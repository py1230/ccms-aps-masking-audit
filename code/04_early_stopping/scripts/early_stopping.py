import run_supplement as r
import numpy as np,time
from sklearn.ensemble import HistGradientBoostingClassifier
T=time.time();df,meta=r.load_aps_csv(r.DATA/'aps/aps_failure_training_set.csv');X,y=r.split_features(df);sp=r.dev_split(y,seed=11,stratify=True);tr,va=sp['train'],sp['val'];x,yy=X[tr],y[tr];imp=r.MedianImputer().fit(x);ev={k:v[0] for k,v in r.build_eval_inputs(X[va]).items()};q,cnt,_=r.estimate_class_rates(x,yy);bv=r.class_blind_rates(q,cnt,.25)
ps={};info={}
for cfg in ['M3','C025','B025']:
 runs=[]
 for ms in ([None] if cfg=='M3' else r.MASK):
  xm=x if cfg=='M3' else (r.mask_class_conditional(x,yy,.25,seed=ms)[0] if cfg=='C025' else r.mask_class_blind(x,bv,seed=ms)[0])
  m=HistGradientBoostingClassifier(random_state=2026,early_stopping=False,max_iter=100);m.fit(r.matrix(xm,imp),yy)
  runs.append({k:m.predict_proba(r.matrix(v,imp))[:,1] for k,v in ev.items()})
 ps[cfg]={k:np.mean([z[k] for z in runs],0) for k in ev};info[cfg]=r.metrics(y[va],ps[cfg]);print(cfg,'done',time.time()-T,flush=True)
r.save('early_stopping_sensitivity.json',{'environment':r.ENV,'seed':11,'alpha':.25,'early_stopping':False,'max_iter':100,'metrics':info,'seconds':time.time()-T})
np.savez_compressed(r.RES/'early_stopping.npz',y=y[va],**{c+'__'+k:p for c,z in ps.items() for k,p in z.items()})
