"""Prepare a fresh self-contained run; copy no existing results or freeze receipts."""
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination',type=Path);args=ap.parse_args()
    dest=args.destination.expanduser().resolve()
    if dest.exists():raise FileExistsError(f'Refuse existing directory: {dest}')
    names=['README.md','PROTOCOL.md','config.json','run_simulation.py','launch_simulation.py','freeze_run.py',
           'analyze_results.py','verify_results.py','prepare_repeat.py','environment_packages.txt',
           'audit/runner_code_review.md','audit/statistical_design_review.md','audit/generator_source_audit.md']
    names += [str(p.relative_to(HERE)) for folder in ['source','historical'] for p in sorted((HERE/folder).glob('*.py'))]
    for name in names:
        if not (HERE/name).is_file():raise FileNotFoundError(HERE/name)
    dest.mkdir(parents=True)
    for name in names:
        out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,out)
        assert sha(out)==sha(HERE/name)
    record={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(HERE),'destination':str(dest),
            'copied_files':{name:sha(dest/name) for name in names},'results_copied':False,'freeze_copied':False,
            'review_status':'Existing source review copied as historical evidence, not a new independent scientific review',
            'required_before_execution':'Inspect unchanged scientific code/config and copied review; run new smoke, then freeze/check. Any scientific edits require new independent review.'}
    (dest/'repeat_preparation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
