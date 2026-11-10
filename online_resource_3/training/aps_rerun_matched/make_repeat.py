"""Create an isolated repeat directory from frozen inputs; never starts training."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('destination', type=Path)
    ap.add_argument('--data-dir', type=Path, required=True)
    a=ap.parse_args(); dest=a.destination.resolve(); data=a.data_dir.resolve()
    if dest.exists(): raise SystemExit('Destination must not exist; choose a new run directory.')
    fr=json.loads((HERE/'freeze.json').read_text())
    # Check local package files, independent of original absolute file locations.
    source_files={}
    for name,e in fr['files'].items():
        if name in ['aps_failure_training_set.csv','aps_failure_test_set.csv']:
            source=data/name
        else:
            rel=Path(name)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError(name)
            source=HERE/rel
        assert sha(source)==e['sha256'],f'Input differs from frozen package: {source}'
        source_files[name]=source
    dest.mkdir(parents=True,exist_ok=False)
    files={}
    for name,source in source_files.items():
        if name.endswith('.csv'):
            target=source
        else:
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        files[name]={'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size}
    cfg=json.loads((dest/'config.json').read_text())
    cfg['train_csv']=str(data/'aps_failure_training_set.csv')
    cfg['test_csv']=str(data/'aps_failure_test_set.csv')
    (dest/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    files['config.json'].update(sha256=sha(dest/'config.json'),bytes=(dest/'config.json').stat().st_size)
    new_fr={'frozen_utc':datetime.now(timezone.utc).isoformat(),'files':files,
            'parent_freeze_sha256':sha(HERE/'freeze.json'),'historical_results_previously_seen':True,
            'new_full_model_run_started':False,'note':'Identical scientific configuration; absolute paths relocated for a new independent execution.'}
    (dest/'freeze.json').write_text(json.dumps(new_fr,ensure_ascii=False,indent=2))
    for name in ['make_repeat.py','compare_historical.py','README.md']:
        if (HERE/name).exists():shutil.copy2(HERE/name,dest/name)
    print(dest)
    print('Prepared only. Run run_aps.py --check, then run_aps.py, then verify_results.py in this directory.')
if __name__=='__main__':main()
