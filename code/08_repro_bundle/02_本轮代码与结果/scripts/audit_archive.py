import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(str(Path(__file__).resolve().parents[3] / '03_原始项目归档/CCMS核查_2026-10-31'));OUT=Path(__file__).resolve().parents[1]/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
FAM={'S0':['S0'],'S1_30':['S1_30_101','S1_30_202','S1_30_303'],'S2':['S2_1101','S2_1202','S2_1303'],'S3':['S3_2101','S3_2202','S3_2303']}
raw=json.loads((ROOT/'records/aps_formal_20261031T142351.json').read_text());test=json.loads((ROOT/'records/aps_testset_20261031T145423.json').read_text())
freeze=json.loads((ROOT/'records/protocol_v2_freeze_20261031T141021.json').read_text())
inv={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file()}
out={'archive_inventory':inv,'freeze_check':{n:{'expected':h,'actual':inv.get(n),'match':h==inv.get(n)} for n,h in freeze['files'].items()},'corrected_selection':[],'historical_test_config_n':{},'historical_environment':{}}
for p in (ROOT/'records').glob('*.json'):
 x=json.loads(p.read_text())
 if 'sklearn_version' in x:out['historical_environment'][p.name]=x['sklearn_version']
for s in raw['splits']:
 base=s['metrics']['M3'];worst=min(base[k]['ap'] for k in ['S0','S1_10_101','S1_10_202','S1_10_303']);row={'seed':s['seed'],'arms':{},'test_metrics':{}}
 ts=next(z for z in test['splits'] if z['seed']==s['seed'])
 for arm in ['C','B']:
  cands=[{'alpha':0.,'cfg':'M3','s0_ap':base['S0']['ap'],'worst4':worst,'eligible':True}]+s['selection'][arm]['candidates']
  best=max([v for v in cands if v['eligible']],key=lambda v:(v['worst4'],-v['alpha']))
  row['arms'][arm]={'historical_chosen':s['selection'][arm]['chosen_alpha'],'corrected':best,'all_candidates':cands}
  assert best['cfg'] in ts['metrics']
  row['test_metrics'][arm]={f:float(np.mean([ts['metrics'][best['cfg']][k]['ap'] for k in ks])) for f,ks in FAM.items()}
 out['corrected_selection'].append(row)
for cfg in set(k for s in test['splits'] for k in s['metrics']):out['historical_test_config_n'][cfg]=sum(cfg in s['metrics'] for s in test['splits'])
out['confirmed_findings']=[
 {'id':'A1','severity':'major','finding':'APS alpha=0 only fallback, not competing candidate; corrected C chooses 0 in all five historical splits','source':'scripts/run_aps_formal.py select'},
 {'id':'S1','severity':'major','finding':'SECOM OOF arrays overwritten at each mask seed; retained predictions are final seed9109, not ens3','source':'scripts/run_secom_formal.py store'},
 {'id':'S2','severity':'major','finding':'SECOM formal runner never calls inner_split; fixed alpha all outer train fitted. No fold-specific alpha selection','source':'scripts/run_secom_formal.py'},
 {'id':'T1','severity':'major','finding':'Official test B050 only exists on 3 splits; table builder silently averages available configs under a five-split caption','source':'scripts/make_paper_tables.py mean_table'},
 {'id':'M1','severity':'moderate','finding':'Existing simulation refits imputer on each masked arm, while APS uses fixed original-training imputer','source':'scripts/sim_nuisance.py run_config'},
 {'id':'M2','severity':'moderate','finding':'Simulation reverse delta=-0.4, first-column b=0.2 gives negative q1 and is clipped to0; expected overall missingness is no longer constant','source':'scripts/sim_nuisance.py apply_missing'},
 {'id':'D1','severity':'moderate','finding':'Mechanism ablations/polarity use split7; deletion_stats uses split11; these do not share one split as former table implied','source':'scripts/diag_cb_collapse.py, diag_polarity.py, make_paper_tables.py'},
 {'id':'R1','severity':'moderate','finding':'Formal APS script fits25models per split,125 across five, not25 overall;13reported configs aggregate the mask seed models','source':'scripts/run_aps_formal.py'},
 {'id':'R2','severity':'moderate','finding':'Historical formal APS/SECOM records contain metrics not formal per-sample predictions; only pilot probability arrays archived','source':'records inventory'}]
(OUT/'archive_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('corrected alpha C',[s['arms']['C']['corrected']['alpha'] for s in out['corrected_selection']])
print('test configs n',out['historical_test_config_n'])
print('historical sklearn',out['historical_environment'])
print('archive files',len(inv))
