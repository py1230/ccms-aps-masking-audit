"""Package finalized evidence plus both raw CSVs, then hash every ZIP entry."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime,timezone

P=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def main():
    assert json.loads((P/'verification.json').read_text())['status']=='passed'
    assert json.loads((P/'run_results/summary.json').read_text())['status']=='complete'
    cfg=json.loads((P/'config.json').read_text())
    archive=P.parent/'APS主分析重跑完整材料_20261108.zip'
    assert not archive.exists(), 'Package exists; use a new version rather than overwrite.'
    entries={str(f.relative_to(P)):f for f in sorted(P.rglob('*')) if f.is_file()
             and '__pycache__' not in f.parts and f not in [P/n for n in ['manifest.json','package_verification.json','package_stdout.log','package_stderr.log']]}
    for key in ['train_csv','test_csv']:
        f=Path(cfg[key]);entries['raw_data/'+f.name]=f
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'root':P.name,
              'note':'raw_data entries are included directly from project CSVs in the ZIP; no extra on-disk copy is created. Packaging process logs and its final verification record are excluded to avoid self-referential hashes.',
              'files':{name:{'sha256':digest(f),'bytes':f.stat().st_size} for name,f in entries.items()}}
    (P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    entries['manifest.json']=P/'manifest.json'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,f in entries.items():z.write(f,P.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        saved=json.loads(z.read(P.name+'/manifest.json'))
        assert set(z.namelist())=={P.name+'/'+n for n in entries}
        for name,e in saved['files'].items():
            b=z.read(P.name+'/'+name)
            assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
    result={'status':'passed','archive':str(archive),'archive_sha256':digest(archive),
            'archive_bytes':archive.stat().st_size,'verified_payload_files':len(manifest['files']),
            'manifest_included':True,'finished_utc':datetime.now(timezone.utc).isoformat()}
    (P/'package_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
