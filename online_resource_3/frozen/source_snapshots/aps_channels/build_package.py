"""Package completed, verified outputs and raw data; verify every archived payload hash."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--zip-path',required=True);a=p.parse_args()
    report=HERE/a.report;target=Path(a.zip_path).resolve()
    assert report.is_file() and report.stat().st_size>0
    assert not target.exists() and not (HERE/'manifest.json').exists() and not (HERE/'package_verification.json').exists()
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    assert verification['checked']['models']==855
    for key, rel in [('config_sha256','config.json'),('freeze_sha256','freeze.json'),('runner_sha256','run_channels.py'),('verifier_sha256','verify_results.py')]:
        assert verification[key]==sha(HERE/rel),f'Stale verification binding: {key}'
    for ch, row in verification.get('channel_reports',{}).items():
        assert row['sha256']==sha(HERE/row['file']),f'Changed channel report: {ch}'
    launcher=json.loads((HERE/'launcher_receipt.json').read_text())
    for ch in ['M2','M3','MO']:
        s=json.loads((HERE/'run_results'/ch/'summary.json').read_text())
        assert s['status']=='complete' and s['fits_completed']==285
        assert launcher['channels'][ch]['returncode']==0
    cfg=json.loads((HERE/'config.json').read_text());freeze=json.loads((HERE/'freeze.json').read_text())
    for name,r in freeze['files'].items():
        fp=Path(r['path']);fp=fp if fp.is_absolute() else HERE/fp
        assert sha(fp)==r['sha256'],f'Frozen payload changed: {name}'
    pairs=[]
    for f in sorted(HERE.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.name.startswith('.'):continue
        if f.suffix in ['.pyc','.zip'] or f.name in ['manifest.json','package_verification.json']:continue
        pairs.append((f,str(f.relative_to(HERE))))
    for k in ['train_csv','test_csv']:
        f=Path(cfg[k]);pairs.append((f,'raw_data/'+f.name))
    names=[n for _,n in pairs];assert len(names)==len(set(names))
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'all payload files, excludes manifest itself and external packaging receipt',
              'files':{n:{'sha256':sha(f),'size_bytes':f.stat().st_size} for f,n in pairs}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f,n in pairs:z.write(f,n)
        z.write(HERE/'manifest.json','manifest.json')
    total_bytes=0
    with zipfile.ZipFile(target) as z:
        assert set(z.namelist())==set(names)|{'manifest.json'}
        assert z.read('manifest.json')==(HERE/'manifest.json').read_bytes()
        for name,row in manifest['files'].items():
            h=hashlib.sha256();size=0
            with z.open(name) as stream:
                for b in iter(lambda:stream.read(1<<20),b''):h.update(b);size+=len(b)
            assert h.hexdigest()==row['sha256'] and size==row['size_bytes'],name
            total_bytes+=size
    receipt={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'zip_path':str(target),
             'zip_sha256':sha(target),'zip_size_bytes':target.stat().st_size,'payload_file_count':len(pairs),
             'payload_size_bytes':total_bytes,'method':'stream-decompressed every ZIP member, checked SHA256 and byte length, matched complete member inventory and manifest bytes',
             'manifest_sha256':sha(HERE/'manifest.json'),'models':855,
             'boundary':'Archive integrity is not experiment provenance or independent retraining.'}
    (HERE/'package_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
