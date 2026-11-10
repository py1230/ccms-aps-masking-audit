"""Prepare an isolated repeat; do not copy models, predictions or old freeze."""
from pathlib import Path
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination');a=ap.parse_args();dest=Path(a.destination).resolve();dest.mkdir(parents=True,exist_ok=False)
    names=['run_resample.py','summarize_results.py','launch_runs.py','make_repeat.py','config.json','PROTOCOL.md','README.md','environment.json','environment_packages.txt','input_sources.json','split_indices.npz','validation_masks.npz','test_masks.npz']
    names += [n for n in ['verify_results.py','summarize_resample_identity.py'] if (HERE/n).is_file()]
    for name in names:shutil.copy2(HERE/name,dest/name)
    for folder in ['source','raw_data','prepared_indices']:shutil.copytree(HERE/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__'))
    fingerprints={}
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            name=str(p.relative_to(dest));h=sha(p);assert h==sha(HERE/name);fingerprints[name]=h
    (dest/'repeat_preparation.json').write_text(json.dumps({'source':str(HERE),'destination':str(dest),'copied_sha256':fingerprints,
        'scope':'same scientific design and bootstrap index plan; inputs only; no old formal/smoke models, predictions, freeze or verification results copied'},ensure_ascii=False,indent=2)+'\n')
    print(dest)
if __name__=='__main__':main()
