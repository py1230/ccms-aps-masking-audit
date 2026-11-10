"""Create a single-use pre-run freeze. Do not overwrite prior freezes."""
import hashlib,json,subprocess,sys
from importlib.metadata import distributions
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text())
assert not (ROOT/'freeze.json').exists()
assert not (ROOT/'run_results').exists()
assert json.loads((ROOT/'smoke_results/smoke.json').read_text())['status']=='passed'
assert (ROOT/'audit/runner_code_review.md').is_file(), 'code review required'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
files={}
paths=[ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'split_indices.npz',ROOT/'run_channels.py',ROOT/'freeze_run.py',ROOT/'launch_channels.py',ROOT/'audit/source_copy_check.json',ROOT/'audit/runner_code_review.md']
paths+=list((ROOT/'source').glob('*.py'))+list((ROOT/'historical').glob('*.py'))
paths += [Path(cfg['train_csv']),Path(cfg['test_csv'])]
for p in paths:
    rel=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    files[rel]={'path':rel,'sha256':sha(p),'size_bytes':p.stat().st_size}
packages=sorted({d.metadata['Name']+'=='+d.version for d in distributions()})
(ROOT/'environment_packages.txt').write_text('\n'.join(packages)+'\n')
files['environment_packages.txt']={'path':'environment_packages.txt','sha256':sha(ROOT/'environment_packages.txt')}
(ROOT/'freeze.json').write_text(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'this retrospective execution only, not preregistration','files':files,'expected_models':855},ensure_ascii=False,indent=2))
print('Frozen',len(files),'files')
