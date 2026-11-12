"""Check the recovered A3 artifacts without importing or running training code."""
from pathlib import Path
import argparse, hashlib, itertools, json, zipfile

HERE=Path(__file__).resolve().parent
HIST=HERE.parent
def sha(data):return hashlib.sha256(data).hexdigest()
def decoded(info):
    try:return info.filename.encode('cp437').decode('utf-8') if not info.flag_bits&0x800 else info.filename
    except (UnicodeEncodeError,UnicodeDecodeError):return info.filename
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():raise FileExistsError(args.output)
    metadata=json.loads((HERE/'provenance.json').read_text())
    archive=HERE/'original_recovery.zip'
    assert sha(archive.read_bytes())==metadata['source_zip_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        fs={'/'.join(decoded(i).split('/')[1:]):z.read(i) for i in z.infolist() if not i.is_dir()}
    for rel,data in fs.items():assert (HERE/'original_export'/rel).read_bytes()==data,rel
    manifest_checks=0
    for line in fs['04_SHA256清单/SHA256SUMS.txt'].decode().splitlines():
        digest,name=line.split(maxsplit=1);name=name.removeprefix('./')
        assert sha(fs[name])==digest,name
        manifest_checks+=1
    records={Path(n).name:json.loads(b) for n,b in fs.items() if '/results_seed/' in n}
    expected=set(itertools.product([11,22,33,44,55],['C','B'],[2026,2027,2028]))
    assert {(v['split'],v['arm'],v['model_seed']) for v in records.values()}==expected
    recovered=HIST/'recovered/new_experiments_2026-11-05'
    for n,b in fs.items():
        if '/results_seed/' in n:assert (recovered/'results_seed'/Path(n).name).read_bytes()==b,n
    code=fs['01_原始文件/run_seed_dispersion.py']
    assert code==(recovered/'run_seed_dispersion.py').read_bytes()
    command=next(b for n,b in fs.items() if n.endswith('block121.txt'))
    embedded=command.split(b"<<'EOF'\n",1)[1].split(b'\nEOF\n',1)[0]+b'\n'
    assert code==embedded
    assert (recovered/'seed.log').read_bytes()==fs['01_原始文件/seed.log']
    lines=fs['01_原始文件/seed.log'].decode().splitlines()
    assert lines[-1]=='SEED_DONE' and len(lines)==31
    for line in lines[:-1]:
        split,arm,ms,ap=line.split()
        assert round(records[f's{split}_{arm}_{ms}.json']['ap'],4)==float(ap)
    deps=json.loads((HERE/'dependency_fingerprints.json').read_text())
    for item in deps:assert sha((HIST/item['path_from_historical_recovery']).read_bytes())==item['sha256']
    terminal=json.loads(fs['02_原始终端回执/bg_9f8d59e842f8.meta.json'])
    assert command.decode().split('\n\n',1)[1].startswith(terminal['command'])
    result={'status':'passed','training_performed':False,'original_export_files':len(fs),'original_manifest_checks':manifest_checks,'AP_records_matching_prior_archive':len(records),'log_rows_matching_AP':30,'runner_matches_creation_command':True,'dependency_fingerprint_checks':len(deps),'scope':'Artifact and scalar-log consistency only; no prediction-level AP recomputation or historical training reproduction.','runtime_versions_recovered':False}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
