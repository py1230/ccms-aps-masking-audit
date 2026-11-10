"""Create a fresh portable rerun folder, without copying results or starting fits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination');p.add_argument('--data-dir',required=True)
    a=p.parse_args();dest=Path(a.destination).resolve();data=Path(a.data_dir).resolve()
    if dest.exists():raise FileExistsError(f'Destination exists: {dest}')
    cfg=json.loads((HERE/'config.json').read_text());fr=json.loads((HERE/'freeze.json').read_text())
    for key in ['train_csv','test_csv']:
        src=Path(cfg[key]);candidate=data/src.name
        found=[r['sha256'] for r in fr['files'].values() if Path(r['path']).name==src.name]
        assert len(found)==1 and sha(candidate)==found[0],f'Data checksum mismatch: {candidate}'
        cfg[key]=str(candidate)
    # Check source scientific bytes against the original execution freeze.
    for rel in ['run_channels.py','freeze_run.py','launch_channels.py','config.json','PROTOCOL.md','split_indices.npz']:
        assert sha(HERE/rel)==fr['files'][rel]['sha256'],f'Frozen source changed: {rel}'
    dest.mkdir(parents=True)
    for folder in ['source','historical']:
        (dest/folder).mkdir()
        for file in (HERE/folder).glob('*.py'):
            rel=f'{folder}/{file.name}'
            assert sha(file)==fr['files'][rel]['sha256']
            shutil.copy2(file,dest/rel)
    for rel in ['run_channels.py','freeze_run.py','launch_channels.py','PROTOCOL.md','split_indices.npz','README.md','make_repeat.py','verify_results.py']:
        shutil.copy2(HERE/rel,dest/rel)
    if (HERE/'scripts').is_dir():
        (dest/'scripts').mkdir()
        for file in (HERE/'scripts').glob('*.py'):shutil.copy2(file,dest/'scripts'/file.name)
    for rel in ['summarize_results.py','build_package.py']:
        if (HERE/rel).exists():shutil.copy2(HERE/rel,dest/rel)
    (dest/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    (dest/'audit').mkdir()
    for rel in ['runner_code_review.md','source_copy_check.json']:
        shutil.copy2(HERE/'audit'/rel,dest/'audit'/rel)
    receipt={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source_directory':str(HERE),
             'source_freeze_sha256':sha(HERE/'freeze.json'),'config_changes':['train_csv','test_csv'],
             'scientific_changes':[], 'status':'prepared_not_run',
             'copied_review_scope':'source review retained for unchanged scientific implementation; config paths relocated, old snapshot hashes describe original source',
             'next':['run_channels.py --smoke','freeze_run.py','run_channels.py --check','launch_channels.py','verify_results.py']}
    (dest/'repeat_preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'prepared':str(dest),'trained':False,'next':'Use the documented matching environment, run --smoke, freeze_run.py, --check, then launch_channels.py'},ensure_ascii=False))
if __name__=='__main__':main()
