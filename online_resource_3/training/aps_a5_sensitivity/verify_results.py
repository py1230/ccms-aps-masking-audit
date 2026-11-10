"""Independent checks: no imports from the generation runner or summarizer."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import ctypes,gzip,hashlib,json,subprocess,time,traceback,gc,resource
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def array_digest(a):return hashlib.sha256(a.dtype.str.encode()+json.dumps(list(a.shape),separators=(',',':')).encode()+a.tobytes(order='C')).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read_raw(p):
 with p.open() as f:
  for i,s in enumerate(f):
   if s.startswith('class,'):break
  else:raise ValueError('header absent')
 d=pd.read_csv(p,skiprows=i,na_values=['na']);assert set(d['class'].unique())=={'pos','neg'}
 return d.iloc[:,1:].to_numpy(dtype=np.float64),(d['class']=='pos').to_numpy(dtype=np.int64),d.columns[1:].to_numpy()
def features(x,med):
 z=np.empty((len(x),2*x.shape[1]),np.float64);missing=np.isnan(x);z[:,:x.shape[1]]=x
 for j in range(x.shape[1]):z[missing[:,j],j]=med[j]
 z[:,x.shape[1]:]=missing;return z
def ap(y,p):
 order=np.argsort(-p,kind='stable');yp=y[order];sp=p[order];last=np.r_[np.flatnonzero(sp[1:]!=sp[:-1]),len(sp)-1];tp=np.cumsum(yp)[last];precision=tp/(last+1);return float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())
def fam(k):return k if k=='S0' else k.rsplit('_',1)[0]
def main():
 out=ROOT/'verification';out.mkdir(exist_ok=False);t0=time.perf_counter();events=out/'events.jsonl';c=json.loads((ROOT/'config.json').read_text());results={'status':'running','independent_of_runner_imports':True}
 def event(x):
  x={'seconds':time.perf_counter()-t0,**x}
  with events.open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
 try:
  frozen=json.loads((ROOT/'freeze.json').read_text());checked=0
  for f,h in frozen['files'].items():assert digest(ROOT/f)==h,f;checked+=1
  results['frozen_files_checked']=checked
  assert json.loads((ROOT/'run_results/complete.json').read_text())['fits']==270
  x,y,names=read_raw(ROOT/'raw_data/aps_failure_training_set.csv');xt,yt,nt=read_raw(ROOT/'raw_data/aps_failure_test_set.csv');assert x.shape==(60000,170) and xt.shape==(16000,170) and yt.sum()==375 and np.array_equal(names,nt)
  subprocess.run(['cc','-O3','-shared','-fPIC',str(ROOT/'source/verify_switch.c'),'-o',str(out/'verify_switch.dylib')],check=True)
  library=ctypes.CDLL(str(out/'verify_switch.dylib'));ptr=ctypes.c_void_p;fn=library.verify_proposals;fn.argtypes=[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_uint32,ptr];fn.restype=ctypes.c_size_t
  chain_count=mask_count=proposal_count=accepted_count=0;chain_results=[]
  for split in c['splits']:
   pr=np.load(ROOT/f'reference/split{split}/preprocessing.npz');ids=pr['train_rows'];assert len(ids)==42000 and len(np.unique(ids))==42000 and np.all((ids>=0)&(ids<60000));assert np.array_equal(names,pr['feature_names']);allowed=(~np.isnan(x[ids])).astype(np.uint8)
   for rs in c['references']:
    reference=np.load(ROOT/f'reference/split{split}/training_masks/C025_seed{rs}.npz')['new_mask'];base=ROOT/f'masks/split{split}/ref{rs}';d=np.load(base/'initial_D/mask_100E.npz')['new_mask'];assert not np.array_equal(reference,d)
    specs=[(base/'initial_D',reference,920000000+split*100000+rs,[100])]+[(base/initial/f'rep{rep}',reference if initial=='C' else d,930000000+split*100000+rs*10+ii*10000000+rep,[5,20,50]) for ii,initial in enumerate(c['initials']) for rep in c['repeats']]
    for directory,initial,seed,steps in specs:
     meta=json.loads((directory/'chain.json').read_text());assert meta['seed']==seed and meta['steps']==steps;assert meta['initial_sha']==array_digest(initial) and meta['reference_sha']==array_digest(reference)
     r,col=np.nonzero(initial);r=r.astype(np.uint32);col=col.astype(np.uint32);saved=np.load(directory/'initial_edges.npz');assert np.array_equal(saved['rows'],r) and np.array_equal(saved['cols'],col);e=len(r);assert e==meta['E'];state=initial.astype(np.uint8);rng=np.random.default_rng(seed);assert rng.bit_generator.state==meta['rng_initial'];offset=0;count=0;checkpoints={s*e:s for s in steps}
     with gzip.open(directory/'accepted_trace.u32.gz','rb') as trace:
      for batch in meta['batches']:
       n=batch['n'];assert batch['start']==offset and n<=200000 and batch['rng_before']==rng.bit_generator.state
       aa=rng.integers(0,e,n,dtype=np.uint32);bb=rng.integers(0,e,n,dtype=np.uint32);stay=rng.integers(0,2,n,dtype=np.uint8);assert array_digest(aa)==batch['a_sha'] and array_digest(bb)==batch['b_sha'] and array_digest(stay)==batch['lazy_sha'];assert rng.bit_generator.state==batch['rng_after']
       rec=np.empty((n,3),np.uint32);na=fn(state.ctypes.data,allowed.ctypes.data,r.ctypes.data,col.ctypes.data,aa.ctypes.data,bb.ctypes.data,stay.ctypes.data,n,state.shape[1],offset,rec.ctypes.data);assert na==batch['accepted'];raw=trace.read(na*12);assert len(raw)==na*12 and raw==rec[:na].astype('<u4',copy=False).tobytes();count+=na;offset+=n
       if offset in checkpoints:
        step=checkpoints[offset];m=np.load(directory/f'mask_{step}E.npz');mask=m['new_mask'];assert np.array_equal(state.astype(bool),mask);assert np.array_equal(mask.sum(0),reference.sum(0)) and np.array_equal(mask.sum(1),reference.sum(1));assert not np.any(mask&~allowed.astype(bool));assert np.array_equal(m['row_deleted'],mask.sum(1)) and np.array_equal(m['col_deleted'],mask.sum(0))
        diag=next(z for z in meta['diagnostics'] if z['step']==step);assert diag['mask_sha']==array_digest(mask) and diag['attempts']==offset and diag['accepted']==count;assert abs(diag['relocated_from_C']-(1-(mask&reference).sum()/e))<1e-14;assert abs(diag['relocated_from_initial']-(1-(mask&initial).sum()/e))<1e-14;mask_count+=1
      assert trace.read(1)==b''
     assert offset==max(steps)*e and rng.bit_generator.state==meta['rng_final'];chain_count+=1;proposal_count+=offset;accepted_count+=count;chain_results.append(dict(path=str(directory.relative_to(ROOT)),seed=seed,E=e,proposals=offset,accepted=count))
    event(dict(event='reference_chains_verified',split=split,reference=rs,chains=chain_count))
  assert chain_count==105 and mask_count==285;results.update(chains_checked=chain_count,checkpoints_checked=mask_count,all_proposals_replayed=proposal_count,accepted_records_replayed=accepted_count);save(out/'chain_checks.json',chain_results)
  del allowed,state,aa,bb,stay,rec,reference,d,initial;gc.collect()
  em=np.load(ROOT/'reference/test_evaluation_masks.npz');conds=em.files;source_scores={};scores={};individual=[];model_count=source_count=0;maxdiff=0.;max_ap_error=0.
  for split in c['splits']:
   pr=np.load(ROOT/f'reference/split{split}/preprocessing.npz');ids=pr['train_rows'];med=np.nanmedian(x[ids],axis=0);med[np.isnan(med)]=0;assert np.array_equal(med,pr['medians'])
   entries=[('source',ROOT/f'reference/split{split}/models/{tag}.joblib',ROOT/f'reference/split{split}/predictions/{tag}__test.npz',tag,None) for tag in ['BASE']+[f'{arm}025_seed{rs}' for arm in ['C','B'] for rs in c['references']]]
   entries += [('new',ROOT/f'run_results/split{split}/{initial}/rep{rep}/{step}E/model_ref{rs}.joblib',ROOT/f'run_results/split{split}/{initial}/rep{rep}/{step}E/predictions_ref{rs}.npz',(initial,rep,step,rs),ROOT/f'masks/split{split}/ref{rs}/{initial}/rep{rep}/mask_{step}E.npz') for initial in c['initials'] for rep in c['repeats'] for step in c['steps'] for rs in c['references']]
   source_probs={};new_probs={}
   for kind,model_path,pred_path,key,mask_path in entries:
    payload=joblib.load(model_path);model=payload['model'];z=np.load(pred_path);assert np.array_equal(z['y'],yt) and np.array_equal(z['row_indices'],np.arange(16000));assert np.array_equal(payload['train_rows'],ids) and np.array_equal(payload['medians'],med)
    if kind=='new':
     initial,rep,step,rs=key;assert (payload['split'],payload['initial'],payload['repeat'],payload['step'],payload['reference_seed'])==(split,initial,rep,step,rs);assert payload['model_seed']==2026 and payload['config_sha']==digest(ROOT/'config.json');assert model.get_params()==c['model_params'];m=np.load(mask_path)['new_mask'];assert array_digest(m)==payload['mask_sha'];xm=x[ids].copy();xm[m]=np.nan;matrix=features(xm,med);assert array_digest(matrix)==payload['training_matrix_sha'] and array_digest(y[ids])==payload['training_labels_sha'];del xm,matrix,m;model_count+=1
    else:source_count+=1
    model_max=0.;ps={}
    for k in conds:
     xe=xt.copy();xe[em[k]]=np.nan;calc=model.predict_proba(features(xe,med))[:,1];stored=z[k];assert stored.shape==(16000,) and np.isfinite(stored).all();diff=float(np.max(np.abs(calc-stored)));assert diff<=1e-14,(model_path,k,diff);model_max=max(model_max,diff);ps[k]=stored.copy();individual.append(dict(split=split,kind=kind,key=str(key),condition=k,AP=ap(yt,stored),prediction_max_error=diff));del xe,calc
    maxdiff=max(maxdiff,model_max)
    if kind=='source':source_probs[key]=ps
    else:new_probs[key]=ps
    del payload,model,z;gc.collect()
   for arm in ['C','B','BASE']:
    keys=['BASE'] if arm=='BASE' else [f'{arm}025_seed{rs}' for rs in c['references']];vals={k:ap(yt,sum(source_probs[tag][k] for tag in keys)/len(keys)) for k in conds};source_scores[split,arm]={g:float(np.mean([v for k,v in vals.items() if fam(k)==g])) for g in ['S0','S1_10','S1_30','S2','S3']}
   for initial in c['initials']:
    for rep in c['repeats']:
     for step in c['steps']:
      ens=np.load(ROOT/f'analysis/ensembles/split{split}/{initial}/rep{rep}/{step}E.npz');assert np.array_equal(ens['y'],yt) and np.array_equal(ens['row_indices'],np.arange(16000));vals={}
      for k in conds:
       probability=sum(new_probs[initial,rep,step,rs][k] for rs in c['references'])/3;assert np.max(np.abs(ens[k]-probability))<=1e-15;vals[k]=ap(yt,probability)
      scores[split,initial,rep,step]={g:float(np.mean([v for k,v in vals.items() if fam(k)==g])) for g in ['S0','S1_10','S1_30','S2','S3']}
   del source_probs,new_probs;gc.collect();event(dict(event='split_models_and_probabilities_verified',split=split,new_models=model_count,source_models=source_count,max_probability_error=maxdiff))
  assert model_count==270 and source_count==35;pd.DataFrame(individual).to_csv(out/'individual_AP.csv',index=False)
  table=pd.read_csv(ROOT/'analysis/by_split.csv');assert len(table)==450
  for r in table.itertuples():
   a=scores[r.split,r.initial,r.repeat,r.step][r.group];expected=[a,source_scores[r.split,'C'][r.group]-a,a-source_scores[r.split,'B'][r.group],a-source_scores[r.split,'BASE'][r.group]]
   for actual,exp in zip([r.AP,r.C_minus_A5,r.A5_minus_B,r.A5_minus_BASE],expected):error=abs(actual-exp);assert error<1e-13;max_ap_error=max(max_ap_error,error)
  repeats={}
  for r in pd.read_csv(ROOT/'analysis/by_repeat.csv').itertuples():
   a=float(np.mean([scores[split,r.initial,r.repeat,r.step][r.group] for split in c['splits']]));refs={arm:float(np.mean([source_scores[split,arm][r.group] for split in c['splits']])) for arm in ['C','B','BASE']};v=dict(AP=a,C_minus_A5=refs['C']-a,A5_minus_B=a-refs['B'],A5_minus_BASE=a-refs['BASE']);repeats[r.initial,r.step,r.repeat,r.group]=v
   for k,e in v.items():assert abs(getattr(r,k)-e)<1e-13
  def check_stats(s,v):
   for k,e in dict(mean=np.mean(v),sd=np.std(v,ddof=1),minimum=np.min(v),maximum=np.max(v)).items():assert abs(s[k]-e)<1e-13
   assert s['n']==3
  summary=json.loads((ROOT/'analysis/summary.json').read_text());assert len(summary)==120
  for s in summary:check_stats(s,[repeats[s['initial'],s['step'],rep,s['group']][s['metric']] for rep in [1,2,3]])
  sensitivity=json.loads((ROOT/'analysis/sensitivity.json').read_text());assert len(sensitivity)==35
  for s in sensitivity:
   left,right=s['comparison'].split(':');g=s['group']
   if right=='D-C':step=int(left[:-1]);values=[repeats['D',step,rep,g]['AP']-repeats['C',step,rep,g]['AP'] for rep in [1,2,3]]
   else:step=int(right.split('E')[0]);values=[repeats[left,step,rep,g]['AP']-repeats[left,20,rep,g]['AP'] for rep in [1,2,3]]
   check_stats(s,values)
  results.update(status='passed',new_models_reloaded=model_count,source_models_reloaded=source_count,probability_arrays_checked=(model_count+source_count)*13,probability_values_checked=(model_count+source_count)*13*16000,max_probability_error=maxdiff,max_group_AP_error=max_ap_error,ensemble_units_checked=90,by_split_rows_checked=450,summary_rows_checked=len(summary),sensitivity_rows_checked=len(sensitivity),seconds=time.perf_counter()-t0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='all proposal streams and accepted traces, legal masks, source/new identity and model replay, ensemble AP and descriptive summaries; internal consistency, not historical provenance or mixing certification');save(out/'result.json',results);event(dict(event='complete',**results))
 except BaseException:
  results.update(status='failed',traceback=traceback.format_exc(),seconds=time.perf_counter()-t0);save(out/'failure.json',results);raise
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
