from pathlib import Path
import json,sys
from importlib.metadata import distributions
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_simulation import check,environment,sha
assert not (HERE/'freeze.json').exists() and not (HERE/'run_results').exists()
cfg=json.loads((HERE/'config.json').read_text());check(cfg,False)
s=HERE/'smoke_results'/f"seed{cfg['smoke_seed']}"
r=json.loads((s/'smoke_reload_check.json').read_text());v=json.loads((HERE/'audit/smoke_independent_verification.json').read_text())
assert r['status']=='passed' and r['models']==20 and r['maximum_error']==0 and v['status']=='passed'
for name,h in r['tested_files'].items():assert sha(HERE/name)==h
(HERE/'environment_packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n')
(HERE/'environment.json').write_text(json.dumps(environment(),indent=2))
names=['PROTOCOL.md','config.json','run_simulation.py','launch_simulation.py','freeze_run.py','analyze_results.py','verify_results.py','environment_packages.txt','environment.json','audit/source_provenance.json','audit/boundary_check.json','audit/smoke_independent_verification.json']+[str(p.relative_to(HERE)) for p in sorted((HERE/'source').glob('*.py'))]+[str((s/'smoke_reload_check.json').relative_to(HERE))]
(HERE/'freeze.json').write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'files':{name:sha(HERE/name) for name in names},'formal_data_generated_before_freeze':False,'formal_performance_viewed_before_freeze':False,'formal_seeds':cfg['data_seeds'],'parent_design_confirmation':'Confirmed label+observed covariate logit mechanism; smoke then freeze/launch authorised in current team message'},indent=2))
print('Frozen scientific inputs after passed independent-implementation smoke')
