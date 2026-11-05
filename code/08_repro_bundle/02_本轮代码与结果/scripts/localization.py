import run_supplement as r
import numpy as np,time
from sklearn.metrics import average_precision_score as AP
X,y=r.split_features(r.load_aps_csv(r.DATA/'aps/aps_failure_training_set.csv')[0]);out={'alpha':.25,'design':'same actual deletion count in each selected column; positive-targeted vs row-class-blind; other cells unchanged','splits':[]}
for seed in r.SEEDS:
 t=time.time();sp=r.dev_split(y,seed=seed,stratify=True);tr,va=sp['train'],sp['val'];x,yy=X[tr],y[tr];imp=r.MedianImputer().fit(x);q,cnt,_=r.estimate_class_rates(x,yy);expected_pos=.25*cnt[1]*(q.max(0)-q[1]);top=np.argsort(-expected_pos)[:8]
 ev={k:v[0] for k,v in r.build_eval_inputs(X[va]).items() if k=='S0' or k.startswith('S1_30')};raw={'C_local':[],'B_local':[]};budgets=[]
 for ms in r.MASK:
  _,info=r.mask_class_conditional(x,yy,.25,seed=ms);dc=np.zeros(x.shape,bool);dc[:,top]=info['new_mask'][:,top]&(yy[:,None]==1)
  db=np.zeros(x.shape,bool);U=np.random.default_rng(ms).random(x.shape)
  for j in top:
   k=int(dc[:,j].sum());idx=np.flatnonzero(~np.isnan(x[:,j]));chosen=idx[np.argsort(U[idx,j],kind='stable')[:k]];db[chosen,j]=True
  assert np.array_equal(dc.sum(0),db.sum(0))
  budgets.append({'mask_seed':ms,'per_column':dc.sum(0).tolist(),'total':int(dc.sum())})
  for cfg,mask in [('C_local',dc),('B_local',db)]:
   xm=x.copy();xm[mask]=np.nan;raw[cfg].append(r.fit_predict(xm,yy,imp,ev))
 ps={c:{k:np.mean([z[k] for z in arr],0) for k in ev} for c,arr in raw.items()};pay={'y':y[va],'validation_rows':va}
 for c,z in ps.items():
  for k,p in z.items():pay[c+'__'+k]=p
 np.savez_compressed(r.RES/f'localization_seed{seed}.npz',**pay)
 out['splits'].append({'seed':seed,'columns':top.tolist(),'budgets':budgets,'metrics':{c:{k:float(AP(y[va],p)) for k,p in z.items()} for c,z in ps.items()},'seconds':time.time()-t});r.save('localization.json',out);print('LOCAL DONE',seed,round(time.time()-t,1),flush=True)
