import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,hashlib,platform,argparse
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score as AP,roc_auc_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
SOURCE=Path(str(Path(__file__).resolve().parents[3] / '03_原始项目归档/CCMS核查_2026-10-31'))
WORK=Path(__file__).resolve().parents[1];RES=WORK/'results'
DATA=Path(str(Path(__file__).resolve().parents[3] / '04_实际使用的数据'))
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'scripts')]
from aps_io import load_aps_csv,split_features,dev_split
from secom_io import load_secom,stratified_kfold,inner_split,select_columns_in_train
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
from pipeline import MedianImputer,_make_model
from run_cb_firstlook import build_eval_inputs,stress_unit
from sim_nuisance import gen_data,base_rates
SEEDS=[11,22,33,44,55];MASK=[9107,9108,9109];ALPHAS=[.25,.5,1.]
def save(name,x):
 (RES/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(x,imp,bb='M3'):
 return x if bb=='M4' else np.hstack([imp.transform(x),np.isnan(x).astype(float)])
def fit_predict(x,y,imp,evals,bb='M3'):
 m=_make_model(2026);m.fit(matrix(x,imp,bb),y)
 return {k:m.predict_proba(matrix(v,imp,bb))[:,1] for k,v in evals.items()}
def metrics(y,ps):return {k:float(AP(y,v)) for k,v in ps.items()}
def select(y,ps):
 cond=['S0','S1_10_101','S1_10_202','S1_10_303'];out={}
 base=AP(y,ps['M3']['S0'])
 for arm in ['C','B']:
  cand=[]
  for a in [0.]+ALPHAS:
   cfg='M3' if a==0 else f'{arm}{round(a*100):03d}'
   vals=[float(AP(y,ps[cfg][c])) for c in cond]
   cand.append({'alpha':a,'config':cfg,'s0':vals[0],'worst4':min(vals),'eligible':bool(vals[0]>=base-.01)})
  out[arm]={'chosen':max([v for v in cand if v['eligible']],key=lambda v:(v['worst4'],-v['alpha'])),'candidates':cand}
 return out

def aps():
 df,meta=load_aps_csv(DATA/'aps/aps_failure_training_set.csv');X,y=split_features(df)
 dt,mt=load_aps_csv(DATA/'aps/aps_failure_test_set.csv');Xt,yt=split_features(dt)
 old=json.loads((SOURCE/'records/aps_formal_20261031T142351.json').read_text())
 out={'environment':ENV,'data':meta,'test':mt,'alpha':.25,'splits':[]}
 for seed in SEEDS:
  t=time.time();sp=dev_split(y,seed=seed,stratify=True);tr,va=sp['train'],sp['val']
  x,yy=X[tr],y[tr];imp=MedianImputer().fit(x)
  ev={f'val__{k}':v[0] for k,v in build_eval_inputs(X[va]).items()}
  ev.update({f'test__{k}':v[0] for k,v in build_eval_inputs(Xt).items()})
  ps={'M3':fit_predict(x,yy,imp,ev)};q,counts,_=estimate_class_rates(x,yy);bvec=class_blind_rates(q,counts,.25)
  delrows=[]
  for arm in ['C','B']:
   arr=[]
   for ms in MASK:
    xm,info=mask_class_conditional(x,yy,.25,seed=ms) if arm=='C' else mask_class_blind(x,bvec,seed=ms)
    arr.append(fit_predict(xm,yy,imp,ev))
    perclass=np.array([info['new_mask'][yy==c].sum(0) for c in [0,1]])
    exp=.25*sum(counts[c]*(q.max(0)-q[c]) for c in [0,1])
    expb=bvec*(~np.isnan(x)).sum(0)
    assert np.allclose(exp,expb,atol=1e-8)
    if arm=='C':
     qi,_,_=estimate_class_rates(xm,yy);posdel=perclass[1];top=np.argsort(-posdel)[:8]
     delrows.append({'mask_seed':ms,'deleted_by_class_col':perclass.tolist(),'expected_per_col':exp.tolist(),'actual_per_col':info['deleted_per_col'].tolist(),'visible_by_class_col':[np.sum(~np.isnan(x[yy==c]),axis=0).tolist() for c in [0,1]],'top8_columns':top.tolist(),'top8_pos_share':float(posdel[top].sum()/max(posdel.sum(),1)),'gap_before':(q[1]-q[0]).tolist(),'gap_after':(qi[1]-qi[0]).tolist()})
    else:delrows.append({'mask_seed':ms,'arm':'B','deleted_by_class_col':perclass.tolist(),'actual_per_col':info['deleted_per_col'].tolist()})
    print(f'APS {seed} {arm} {ms} elapsed {time.time()-t:.1f}s',flush=True)
   ps[arm+'025']={k:np.mean([a[k] for a in arr],0) for k in ev}
  payload={'y_val':y[va],'val_rows':va,'y_test':yt,'train_rows':tr}
  for cfg,v in ps.items():
   for k,p in v.items():payload[cfg+'__'+k]=p
  np.savez_compressed(RES/f'aps_seed{seed}.npz',**payload)
  summ={cfg:{'val':metrics(y[va],{k[5:]:p for k,p in v.items() if k.startswith('val__')}),'test':metrics(yt,{k[6:]:p for k,p in v.items() if k.startswith('test__')})} for cfg,v in ps.items()}
  osplit=next(v for v in old['splits'] if v['seed']==seed)
  drift=max(abs(summ[c]['val'][k]-osplit['metrics'][c][k]['ap']) for c in ps for k in summ[c]['val'])
  out['splits'].append({'seed':seed,'metrics':summ,'deletion_diagnostics':delrows,'max_abs_old_validation_ap_difference':drift,'seconds':time.time()-t})
  save('aps_reconstruction.json',out);print('APS DONE',seed,'max drift',drift,flush=True)

def secom():
 X,y,meta=load_secom(DATA/'secom/secom.data',DATA/'secom/secom_labels.data');out={'environment':ENV,'data':meta,'folds':[]}
 oof={};foldof=np.full(len(y),-1);selections=[]
 for fi,(outertr,te) in enumerate(stratified_kfold(y,5,11)):
  t=time.time();fit,valid=inner_split(y[outertr],.2,7);tr,va=outertr[fit],outertr[valid]
  assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
  keep=select_columns_in_train(X[tr]);x,yy=X[tr][:,keep],y[tr];imp=MedianImputer().fit(x)
  ve={k:v[0] for k,v in build_eval_inputs(X[va][:,keep]).items() if k=='S0' or k.startswith('S1_10')}
  ee={k:v[0] for k,v in build_eval_inputs(X[te][:,keep]).items() if k=='S0' or k.startswith('S1_')}
  ev={f'val__{k}':v for k,v in ve.items()};ev.update({f'test__{k}':v for k,v in ee.items()})
  ps={bb:fit_predict(x,yy,imp,ev,bb) for bb in ['M3','M4']}
  q,cnt,_=estimate_class_rates(x,yy)
  for a in ALPHAS:
   bv=class_blind_rates(q,cnt,a)
   for arm in ['C','B']:
    runs=[]
    for ms in MASK:
     xm,info=mask_class_conditional(x,yy,a,seed=ms) if arm=='C' else mask_class_blind(x,bv,seed=ms)
     runs.append(fit_predict(xm,yy,imp,ev))
    ps[f'{arm}{round(a*100):03d}']={k:np.mean([z[k] for z in runs],0) for k in ev}
   print('SECOM',fi,'alpha',a,'elapsed',round(time.time()-t,1),flush=True)
  sel=select(y[va],{c:{k[5:]:v for k,v in z.items() if k.startswith('val__')} for c,z in ps.items() if c!='M4'})
  for arm in ['C','B']:ps[arm+'_selected']=ps[sel[arm]['chosen']['config']]
  foldof[te]=fi;pay={'y_val':y[va],'y_test':y[te],'train_rows':tr,'val_rows':va,'test_rows':te,'keep_columns':keep}
  for cfg,z in ps.items():
   for k,p in z.items():
    pay[cfg+'__'+k]=p
    if k.startswith('test__'):
     oof.setdefault(cfg,{}).setdefault(k[6:],np.full(len(y),np.nan))[te]=p
  np.savez_compressed(RES/f'secom_fold{fi}.npz',**pay)
  out['folds'].append({'fold':fi,'n_train':len(tr),'n_validation':len(va),'n_test':len(te),'pos_train':int(y[tr].sum()),'pos_validation':int(y[va].sum()),'pos_test':int(y[te].sum()),'n_columns':len(keep),'selection':sel,'metrics':{c:metrics(y[te],{k[6:]:v for k,v in z.items() if k.startswith('test__')}) for c,z in ps.items()},'seconds':time.time()-t})
  save('secom_corrected.json',out)
  print('SECOM DONE',fi,flush=True)
 payload={'y':y,'fold':foldof}
 for c,z in oof.items():
  for k,p in z.items():assert np.isfinite(p).all();payload[c+'__'+k]=p
 np.savez_compressed(RES/'secom_oof.npz',**payload)
 out['pooled']={c:metrics(y,z) for c,z in oof.items()};save('secom_corrected.json',out)

def simulation():
 out={'environment':ENV,'target_gap_multipliers':[1,.75,.5,.25,0,-.25,-.5,-.75,-1],'seeds':[]}
 for ds in [501,502,503,504,505]:
  t=time.time();X,y,meta=gen_data(ds);rng=np.random.default_rng(ds+7);parts=[[],[],[]]
  for c in [0,1]:
   idx=rng.permutation(np.flatnonzero(y==c));i,j=round(.6*len(idx)),round(.8*len(idx))
   for k,z in enumerate([idx[:i],idx[i:j],idx[j:]]):parts[k].extend(z)
  tr,va,te=[np.sort(v) for v in parts];b,delta=base_rates(ds);b[:6]=.4
  def missing(xx,yy,mult,seed):
   pi=yy.mean();q1=b+(1-pi)*delta*mult;q0=b-pi*delta*mult
   assert q1.min()>=0 and q0.min()>=0 and q1.max()<=1 and q0.max()<=1
   qm=np.where(yy[:,None]==1,q1,q0);U=np.random.default_rng(seed).random(xx.shape)
   z=xx.copy();z[U<qm]=np.nan
   return z,{'q1':q1.tolist(),'q0':q0.tolist(),'expected_overall_missing':float((pi*q1+(1-pi)*q0).mean()),'actual_missing':float(np.isnan(z).mean())}
  x,mtrain=missing(X[tr],y[tr],1,ds*10+1);v,mval=missing(X[va],y[va],1,ds*10+2)
  ve={'S0':v};ve.update({f'S1_10_{s}':stress_unit(v,.1,s)[0] for s in [101,202,303]})
  ee={};target_meta={}
  for i,m in enumerate(out['target_gap_multipliers']):ee[f'gap_{m}'],target_meta[f'gap_{m}']=missing(X[te],y[te],m,ds*10+3)
  ev={f'val__{k}':z for k,z in ve.items()};ev.update({f'test__{k}':z for k,z in ee.items()})
  imp=MedianImputer().fit(x);ps={bb:fit_predict(x,y[tr],imp,ev,bb) for bb in ['M3','M4']};q,cnt,_=estimate_class_rates(x,y[tr])
  for a in ALPHAS:
   bv=class_blind_rates(q,cnt,a)
   for arm in ['C','B']:
    runs=[]
    for ms in MASK:
     xm,info=mask_class_conditional(x,y[tr],a,seed=ms) if arm=='C' else mask_class_blind(x,bv,seed=ms)
     runs.append(fit_predict(xm,y[tr],imp,ev))
    ps[f'{arm}{round(a*100):03d}']={k:np.mean([z[k] for z in runs],0) for k in ev}
  sel=select(y[va],{c:{k[5:]:p for k,p in z.items() if k.startswith('val__')} for c,z in ps.items() if c!='M4'})
  for arm in ['C','B']:ps[arm+'_selected']=ps[sel[arm]['chosen']['config']]
  payload={'y_val':y[va],'y_test':y[te],'train_rows':tr,'val_rows':va,'test_rows':te}
  for c,z in ps.items():
   for k,p in z.items():payload[c+'__'+k]=p
  np.savez_compressed(RES/f'simulation_seed{ds}.npz',**payload)
  out['seeds'].append({'data_seed':ds,'source_selection':sel,'train_missing':mtrain,'validation_missing':mval,'target_missing':target_meta,'metrics':{c:metrics(y[te],{k[6:]:p for k,p in z.items() if k.startswith('test__')}) for c,z in ps.items()},'seconds':time.time()-t})
  save('simulation_continuous.json',out);print('SIM DONE',ds,sel['C']['chosen']['alpha'],sel['B']['chosen']['alpha'],round(time.time()-t,1),flush=True)

ENV={'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':2,'script_sha256':sha(__file__),'source_pipeline_sha256':sha(SOURCE/'src/pipeline.py')}
if __name__=='__main__':
 mode=sys.argv[1];{'aps':aps,'secom':secom,'simulation':simulation}[mode]()
