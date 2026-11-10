"""New exploratory chronological SECOM evaluation; never rewrites historical artifacts."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
import scipy
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits, threadpool_info
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def stamp(): return datetime.now(timezone.utc).isoformat()
def medians(x):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);v=np.nanmedian(x,axis=0)
 return np.where(np.isfinite(v),v,0.)
def matrix(x,med): return np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])
def stress(x,rate,seed):
 z=x.copy();z[(~np.isnan(x))&(np.random.default_rng(seed).random(x.shape)<rate)]=np.nan;return z

def validate_inputs(cfg,freeze):
 for name,e in freeze['files'].items():
  path=Path(e['path']);assert path.is_file(),path
  assert sha(path)==e['sha256'],f'Freeze hash mismatch: {name}'
 X=pd.read_csv(cfg['data_path'],sep=r'\s+',header=None,na_values=['NaN']).to_numpy(float)
 lab=pd.read_csv(cfg['label_path'],sep=r'\s+',header=None)
 assert X.shape==(1567,590) and lab.shape==(1567,2)
 assert set(lab.iloc[:,0].unique())=={-1,1}
 y=(lab.iloc[:,0].to_numpy()==1).astype(int)
 times=pd.to_datetime(lab.iloc[:,1],format='%d/%m/%Y %H:%M:%S',errors='raise').to_numpy()
 assert not pd.isna(times).any()
 assert np.isfinite(X[~np.isnan(X)]).all()
 with np.load(HERE/'split_indices.npz',allow_pickle=False) as z: inds={k:z[k].copy() for k in z.files}
 for w in [1,2,3]:
  a,b,c,d=[inds[f'window{w}_{part}'] for part in ['inner_fit','validation','refit_train','test']]
  assert len(set(a)&set(b))==0 and len(set(c)&set(d))==0
  assert np.array_equal(np.sort(np.r_[a,b]),np.sort(c))
  assert times[a].max()<times[b].min() and times[c].max()<times[d].min()
  for v in [a,b,c,d]: assert len(np.unique(y[v]))==2
 tests=np.concatenate([inds[f'window{w}_test'] for w in [1,2,3]])
 assert len(tests)==len(np.unique(tests))
 return X,y,times,inds

def run():
 cfg=json.loads((HERE/'config.json').read_text());freeze=json.loads((HERE/'freeze.json').read_text())
 X,y,times,inds=validate_inputs(cfg,freeze)
 out=HERE/'run_results';out.mkdir(exist_ok=False)
 started=stamp();t0=time.perf_counter();fit_total=0
 threadpool_limits(limits=2)
 params=cfg['model_params'];defaults=HistGradientBoostingClassifier(**params).get_params()
 env={'started_utc':started,'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__,'threadpool_info':threadpool_info(),'explicit_model_params':params,'effective_all_model_params':defaults,'freeze_sha256':sha(HERE/'freeze.json')}
 dump(out/'environment.json',env)
 def log(obj):
  obj={'time_utc':stamp(),**obj}
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
  print(json.dumps(obj,ensure_ascii=False,allow_nan=False),flush=True)
 log({'event':'start','expected_fits':114})
 result={'design':'new_exploratory_chronological_SECOM','config_sha256':sha(HERE/'config.json'),'freeze_sha256':sha(HERE/'freeze.json'),'windows':[],'inference':'descriptive only; no pooled significance or independent-window inference'}
 def stage(w,stage_name,train_rows,eval_rows,rate):
  nonlocal fit_total
  dest=out/f'window{w}'/stage_name;dest.mkdir(parents=True,exist_ok=False)
  x,yy=X[train_rows],y[train_rows];xe=X[eval_rows];ye=y[eval_rows]
  med=medians(x);q,counts,miss=estimate_class_rates(x,yy)
  evals={'S0':xe,**{f'S1_{int(rate*100)}_{s}':stress(xe,rate,s) for s in cfg['eval_mask_seeds']}}
  emats={name:matrix(v,med) for name,v in evals.items()}
  predictions={'y':ye,'row_indices':eval_rows,'train_rows':train_rows};metrics={};diagnostics=[]
  np.savez_compressed(dest/'preprocessing.npz',medians=med,q=q,class_counts=counts,missing_counts=miss,train_rows=train_rows,eval_rows=eval_rows,feature_indices=np.arange(X.shape[1]))
  def fit(cfg_name,masked,mask_seed,alpha,arm,new_mask):
   nonlocal fit_total
   tag=cfg_name if mask_seed is None else f'{cfg_name}_seed{mask_seed}'
   model=HistGradientBoostingClassifier(**params);start=time.perf_counter()
   model.fit(matrix(masked,med),yy);elapsed=time.perf_counter()-start;fit_total+=1
   assert model.n_iter_==params['max_iter']
   probs={k:model.predict_proba(v)[:,1] for k,v in emats.items()}
   for k,p in probs.items():
    assert p.shape==ye.shape and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    predictions[f'{tag}__{k}']=p
   model_path=dest/(tag+'.joblib')
   joblib.dump({'model':model,'medians':med,'feature_indices':np.arange(X.shape[1]),'training_rows':train_rows,'arm':arm,'alpha':alpha,'mask_seed':mask_seed,'model_seed':params['random_state'],'config_sha256':sha(HERE/'config.json')},model_path,compress=3)
   if new_mask is not None:
    assert not np.any(new_mask & np.isnan(x))
    np.savez_compressed(dest/(tag+'_training_mask.npz'),new_mask=new_mask)
   log({'event':'fit','window':w,'stage':stage_name,'config':tag,'fits_completed':fit_total,'fit_seconds':elapsed,'elapsed_seconds':time.perf_counter()-t0,'n_iter':model.n_iter_,'model_sha256':sha(model_path)})
   return probs
  base=fit('M3',x,None,0.,'shared_zero',None)
  all_probs={'M3':base}
  for alpha in cfg['alphas'][1:]:
   bv=class_blind_rates(q,counts,alpha)
   expected_c=alpha*(counts[:,None]*(q.max(axis=0)[None,:]-q)).sum(axis=0)
   expected_b=bv*(~np.isnan(x)).sum(axis=0)
   assert np.allclose(expected_c,expected_b,rtol=1e-12,atol=1e-9)
   for arm in ['C','B']:
    name=f'{arm}{round(alpha*100):03d}';runs=[]
    for seed in cfg['train_mask_seeds']:
     # Each C/B call starts the same generator and shape: paired common uniforms.
     xm,info=mask_class_conditional(x,yy,alpha,seed=seed) if arm=='C' else mask_class_blind(x,bv,seed=seed)
     nm=info['new_mask']
     assert np.array_equal(np.isnan(xm),np.isnan(x)|nm)
     diagnostics.append({'config':name,'mask_seed':seed,'expected_column_deletions':expected_c.tolist(),'actual_column_deletions':nm.sum(axis=0).tolist(),'actual_class_column_deletions':[nm[yy==v].sum(axis=0).tolist() for v in [0,1]],'actual_row_deletions':nm.sum(axis=1).tolist(),'budget_max_abs_error':float(np.max(np.abs(expected_c-expected_b)))})
     runs.append(fit(name,xm,seed,alpha,arm,nm))
    all_probs[name]={k:np.mean([r[k] for r in runs],axis=0) for k in evals}
  for name,probs in all_probs.items():
   metrics[name]={k:float(average_precision_score(ye,p)) for k,p in probs.items()}
   for k,p in probs.items():predictions[f'{name}__{k}']=p
  np.savez_compressed(dest/'predictions.npz',**predictions)
  dump(dest/'metrics.json',metrics);dump(dest/'deletion_diagnostics.json',diagnostics)
  return metrics
 for w in [1,2,3]:
  wr=out/f'window{w}';a,b,c,d=[inds[f'window{w}_{part}'] for part in ['inner_fit','validation','refit_train','test']]
  val=stage(w,'selection',a,b,.1)
  selections={};candidates={}
  for arm in ['C','B']:
   rows=[]
   for alpha in cfg['alphas']:
    name='M3' if alpha==0 else f'{arm}{round(alpha*100):03d}'
    vals=val[name];rows.append({'alpha':alpha,'config':name,'s0':vals['S0'],'worst4':min(vals.values()),'eligible':bool(vals['S0']>=val['M3']['S0']-cfg['s0_tolerance'])})
   chosen=max([r for r in rows if r['eligible']],key=lambda z:(z['worst4'],-z['alpha']))
   selections[arm]=chosen;candidates[arm]=rows
  selection={'chosen':selections,'candidates':candidates,'selected_before_test_fit':True,'selection_timestamp_utc':stamp()}
  dump(wr/'selection.json',selection);log({'event':'selection_frozen','window':w,'selected_alphas':{a:z['alpha'] for a,z in selections.items()},'selection_sha256':sha(wr/'selection.json')})
  # Full outer past data refit; validation rows now enter fitting only after selection is fixed.
  test=stage(w,'refit_test',c,d,.3)
  report={}
  for name,vals in test.items():report[name]={'S0':vals['S0'],'S1_30_mean_ap':float(np.mean([vals[f'S1_30_{s}'] for s in cfg['eval_mask_seeds']]))}
  for arm in ['C','B']:report[arm+'_selected']=dict(report[selections[arm]['config']])
  comparisons={label:{cond:report[ca][cond]-report[cb][cond] for cond in ['S0','S1_30_mean_ap']} for label,ca,cb in [('fixed_C_minus_B','C025','B025'),('selected_C_minus_B','C_selected','B_selected')]}
  row={'window':w,'sample_counts':{name:{'n':int(len(ix)),'positive':int(y[ix].sum()),'negative':int(len(ix)-y[ix].sum()),'time_min':str(times[ix].min()),'time_max':str(times[ix].max())} for name,ix in [('inner_fit',a),('validation',b),('refit_train',c),('test',d)]},'selection':selection,'test_ap':report,'contrasts':comparisons}
  result['windows'].append(row);dump(out/'summary.json',result);log({'event':'window_complete','window':w,'fits_completed':fit_total})
 assert fit_total==114,fit_total
 result['descriptive_equal_window_mean']={name:{cond:float(np.mean([r['test_ap'][name][cond] for r in result['windows']])) for cond in ['S0','S1_30_mean_ap']} for name in result['windows'][0]['test_ap']}
 result['descriptive_equal_window_contrasts']={name:{cond:float(np.mean([r['contrasts'][name][cond] for r in result['windows']])) for cond in ['S0','S1_30_mean_ap']} for name in ['fixed_C_minus_B','selected_C_minus_B']}
 result.update({'fits_completed':fit_total,'finished_utc':stamp(),'wall_seconds':time.perf_counter()-t0,'status':'complete'})
 dump(out/'summary.json',result)
 log({'event':'complete','fits_completed':fit_total,'wall_seconds':result['wall_seconds']})
 manifest={str(p.relative_to(HERE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in str(p)}
 dump(HERE/'manifest.json',manifest)
 print(json.dumps(result['descriptive_equal_window_contrasts'],indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 if args.check:
  cfg=json.loads((HERE/'config.json').read_text());fr=json.loads((HERE/'freeze.json').read_text());X,y,t,ix=validate_inputs(cfg,fr)
  print(json.dumps({'status':'preflight_passed','shape':X.shape,'positives':int(y.sum()),'model_params':HistGradientBoostingClassifier(**cfg['model_params']).get_params(),'run_results_exists':(HERE/'run_results').exists()},default=str,indent=2))
 else:run()
