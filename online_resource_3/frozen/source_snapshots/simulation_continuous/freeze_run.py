"""Freeze reviewed scientific code, protocol, analysis, environment and smoke receipt."""
from pathlib import Path
import hashlib,json,subprocess,sys
from importlib.metadata import distributions
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_simulation import check,environment,sha

def main():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run_results').exists()
    cfg=json.loads((HERE/'config.json').read_text());check(cfg,False)
    smoke=HERE/'smoke_results'/f"seed{cfg['smoke_seed']}"/'smoke_reload_check.json'
    sr=json.loads(smoke.read_text())
    assert sr['status']=='passed' and sr['models']==20 and sr['maximum_error']==0 and sr['data_seed']==cfg['smoke_seed']
    result=json.loads((smoke.parent/'result.json').read_text())
    assert result['status']=='complete' and result['fits_completed']==20 and result['data_seed']==cfg['smoke_seed'] and result['smoke']
    for name,h in sr['tested_files'].items():assert sha(HERE/name)==h,('smoke code changed',name)
    review=HERE/'audit/runner_code_review.md'
    assert 'Status: PASSED' in review.read_text(),'Require explicit passing independent review'
    env='\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n'
    (HERE/'environment_packages.txt').write_text(env)
    (HERE/'environment.json').write_text(json.dumps(environment(),indent=2))
    names=['PROTOCOL.md','config.json','run_simulation.py','launch_simulation.py','freeze_run.py','analyze_results.py','environment_packages.txt','environment.json']
    names += [str(p.relative_to(HERE)) for folder in ['source','historical'] for p in sorted((HERE/folder).glob('*.py'))]
    names += [str(smoke.relative_to(HERE))]
    names += ['audit/runner_code_review.md','audit/statistical_design_review.md','audit/generator_source_audit.md']
    frozen={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'files':{name:sha(HERE/name) for name in names},
            'formal_data_generated_before_freeze':False,'historical_results_pooled':False,
            'smoke_seed':cfg['smoke_seed'],'formal_seeds':cfg['data_seeds']}
    (HERE/'freeze.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps(frozen,indent=2))
if __name__=='__main__':main()
