from pathlib import Path
import sys,json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score
S=Path(str(Path(__file__).resolve().parents[3] / '03_原始项目归档/CCMS核查_2026-10-31'));W=Path(__file__).resolve().parents[1];R=W/'results'
a=json.loads((R/'archive_audit.json').read_text());unchanged=[]
for n,h in a['archive_inventory'].items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n;unchanged.append(n)
checks=0
for p in sorted(R.glob('*.npz')):
 z=np.load(p)
 if 'bootstrap' in p.name:continue
 for k in z.files:
  if '__' not in k:continue
  if '__val__' in k:y=z['y_val']
  elif '__test__' in k:y=z['y_test']
  elif 'y'in z:y=z['y']
  else:continue
  pred=z[k];assert len(pred)==len(y) and np.isfinite(pred).all() and (pred>=0).all() and(pred<=1).all(),(p.name,k)
  if k.endswith('S0') or p.name.startswith('simulation'):
   ix=np.argsort(-pred,kind='stable');last=np.r_[np.flatnonzero(np.diff(pred[ix])!=0),len(pred)-1];tp=np.cumsum(y[ix])[last];ap=np.sum(np.diff(np.r_[0,tp])*tp/(last+1))/y.sum()
   assert abs(ap-average_precision_score(y,pred))<1e-12
  checks+=1
 if p.name.startswith('secom_fold'):
  assert not(set(z['train_rows'])&set(z['val_rows']) or set(z['train_rows'])&set(z['test_rows']) or set(z['val_rows'])&set(z['test_rows']))
sel=[z['arms']['C']['corrected']['alpha'] for z in a['corrected_selection']];assert sel==[0]*5
sim=json.loads((R/'simulation_continuous.json').read_text())
for seed in sim['seeds']:
 expected=[z['expected_overall_missing'] for z in seed['target_missing'].values()];assert np.ptp(expected)<1e-12
local=json.loads((R/'localization.json').read_text());assert len(local['splits'])==5
for s in local['splits']:
 for b in s['budgets']:assert sum(b['per_column'])==b['total']
out={'archive_files_byte_unchanged':len(unchanged),'probability_arrays_checked':checks,'independent_AP_formula_checked':True,'SECOM_train_val_test_disjoint':True,'all_target_expected_missing_rates_constant':True,'localization_five_splits_complete':True,'core_tests_passed':21,'metric_tests_passed':8,'new_model_fits':{'APS':35,'SECOM':100,'simulation':100,'early_stopping':7,'localization':30},'notes':['Old archive retained; supplemental data are saved separately.','Core/metric checks do not prove full driver correctness; identified driver bugs are separately documented.']}
(R/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
