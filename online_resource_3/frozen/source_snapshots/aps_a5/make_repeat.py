"""Prepare an isolated deterministic repeat; never copy completed results."""
from pathlib import Path
import argparse,shutil,json,hashlib
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination');a=ap.parse_args();dest=Path(a.destination).resolve();dest.mkdir(parents=True,exist_ok=False)
    files=['run_a5.py','summarize_results.py','make_repeat.py','config.json','PROTOCOL.md','README.md','source_config.json','split_indices.npz','reference_sources.json','pilot_decision.json']
    files += [n for n in ['verify_results.py','verify_switch.c','launch_runs.py','mask_structure_summary.py','environment_packages.txt'] if (HERE/n).is_file()]
    for name in files:shutil.copy2(HERE/name,dest/name)
    for folder in ['source','raw_data','reference']:
        shutil.copytree(HERE/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.dylib'))
    check={}
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            relative=p.relative_to(dest);assert sha(p)==sha(HERE/relative);check[str(relative)]=sha(p)
    (dest/'repeat_preparation.json').write_text(json.dumps({'source':str(HERE),'destination':str(dest),'copied_sha256':check,
        'scope':'same frozen scientific configuration; no old masks/results/freeze copied; compile, regenerate all masks, verify then freeze before fitting'},ensure_ascii=False,indent=2)+'\n')
    print(dest)
if __name__=='__main__':main()
