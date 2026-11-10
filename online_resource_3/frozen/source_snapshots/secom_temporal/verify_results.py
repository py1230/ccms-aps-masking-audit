"""Post-run independent artifact reconciliation; no fitting and no result alteration."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib,sys,warnings
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
threadpool_limits(2)
cfg=json.loads((HERE/'config.json').read_text());out=HERE/'run_results'
summary=json.loads((out/'summary.json').read_text());assert summary['status']=='complete' and summary['fits_completed']==114
X=pd.read_csv(cfg['data_path'],sep=r'\s+',header=None).to_numpy(float)
lab=pd.read_csv(cfg['label_path'],sep=r'\s+',header=None);y=(lab[0].to_numpy()==1).astype(int)
models_checked=0;mask_checked=0;ap_checked=0;max_ap_error=0.;max_prediction_error=0.
def ap_independent(y,p):
 # AP = sum threshold-wise increments in recall times precision, grouping exact ties.
 order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
 ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
 tp=np.cumsum(ys)[ends];precision=tp/(ends+1);recall=tp/y.sum()
 return float(np.sum(np.diff(np.r_[0.,recall])*precision))
def mat(x,med):return np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])
for w in [1,2,3]:
 for stage,rate in [('selection',.1),('refit_test',.3)]:
  d=out/f'window{w}'/stage
  with np.load(d/'predictions.npz',allow_pickle=False) as z: p={k:z[k] for k in z.files}
  with np.load(d/'preprocessing.npz',allow_pickle=False) as z: pre={k:z[k] for k in z.files}
  xx=X[p['train_rows']];xe=X[p['row_indices']];yy=y[p['train_rows']]
  assert np.array_equal(p['y'],y[p['row_indices']]);assert np.array_equal(pre['train_rows'],p['train_rows'])
  assert np.array_equal(pre['feature_indices'],np.arange(590))
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(xx,axis=0)
  med=np.where(np.isfinite(med),med,0.);assert np.array_equal(med,pre['medians'])
  q,counts,miss=estimate_class_rates(xx,yy)
  assert np.array_equal(q,pre['q']) and np.array_equal(counts,pre['class_counts']) and np.array_equal(miss,pre['missing_counts'])
  evals={'S0':xe}
  for seed in cfg['eval_mask_seeds']:
   z=xe.copy();z[(~np.isnan(xe))&(np.random.default_rng(seed).random(xe.shape)<rate)]=np.nan;evals[f'S1_{int(100*rate)}_{seed}']=z
  emats={k:mat(v,med) for k,v in evals.items()}
  metrics=json.loads((d/'metrics.json').read_text());diag=json.loads((d/'deletion_diagnostics.json').read_text())
  for name in metrics:
   for cond in evals:
    if name!='M3':
     ens=np.mean([p[f'{name}_seed{s}__{cond}'] for s in cfg['train_mask_seeds']],axis=0)
     assert np.array_equal(ens,p[f'{name}__{cond}'])
    av=ap_independent(p['y'],p[f'{name}__{cond}']);err=abs(av-metrics[name][cond]);max_ap_error=max(max_ap_error,err);assert err<1e-12;ap_checked+=1
  for j in d.glob('*.joblib'):
   model=joblib.load(j);assert np.array_equal(model['training_rows'],p['train_rows']) and np.array_equal(model['medians'],med)
   for k,v in cfg['model_params'].items():assert model['model'].get_params()[k]==v,(k,j)
   for cond,m in emats.items():
    score=model['model'].predict_proba(m)[:,1];err=float(np.max(np.abs(score-p[f'{j.stem}__{cond}'])));max_prediction_error=max(max_prediction_error,err);assert err<1e-12
   models_checked+=1
  for row in diag:
   name=row['config'];alpha=int(name[1:])/100;seed=row['mask_seed'];arm=name[0]
   xm,info=mask_class_conditional(xx,yy,alpha,seed=seed) if arm=='C' else mask_class_blind(xx,class_blind_rates(q,counts,alpha),seed=seed)
   expected=info['new_mask']
   with np.load(d/f'{name}_seed{seed}_training_mask.npz') as z:actual=z['new_mask']
   assert np.array_equal(expected,actual)
   assert np.array_equal(actual.sum(axis=0),row['actual_column_deletions'])
   assert np.array_equal(actual.sum(axis=1),row['actual_row_deletions'])
   assert np.array_equal([actual[yy==v].sum(axis=0) for v in [0,1]],row['actual_class_column_deletions'])
   mask_checked+=1
 selection=json.loads((out/f'window{w}'/'selection.json').read_text());v=json.loads((out/f'window{w}'/'selection/metrics.json').read_text());tm=json.loads((out/f'window{w}'/'refit_test/metrics.json').read_text())
 for arm in ['C','B']:
  candidates=[]
  for alpha in cfg['alphas']:
   name='M3' if alpha==0 else f'{arm}{round(alpha*100):03d}'
   if v[name]['S0']>=v['M3']['S0']-cfg['s0_tolerance']:candidates.append((min(v[name].values()),-alpha,name,alpha))
  chosen=max(candidates);assert chosen[2]==selection['chosen'][arm]['config'] and chosen[3]==selection['chosen'][arm]['alpha']
 for name in ['M3','C025','B025','C050','B050','C100','B100','C_selected','B_selected']:
  src=selection['chosen'][name[0]]['config'] if name.endswith('_selected') else name
  vals={'S0':tm[src]['S0'],'S1_30_mean_ap':float(np.mean([tm[src][f'S1_30_{s}'] for s in cfg['eval_mask_seeds']]))}
  assert vals==summary['windows'][w-1]['test_ap'][name]
assert models_checked==114 and mask_checked==108 and ap_checked==168,(models_checked,mask_checked,ap_checked)
report={'status':'passed','models_reloaded_and_repredicted':models_checked,'training_masks_regenerated':mask_checked,'AP_cells_recomputed_by_independent_tie_aware_formula':ap_checked,'selection_replays':6,'max_abs_AP_error':max_ap_error,'max_abs_reloaded_prediction_error':max_prediction_error,'scope':'new run artifacts only; no historical provenance or generalization claim'}
(HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,indent=2))
