import io,json,hashlib,sys,zipfile,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,sklearn
from sklearn.model_selection import StratifiedGroupKFold
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
from run_cb_firstlook import build_eval_inputs

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def prepare():
 out=HERE/'prepared';out.mkdir(exist_ok=False)
 zpath=HERE/'data/uci447_legacy.zip'
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  profile=np.loadtxt(z.open('profile.txt'),dtype=int)
  sensors=[f'PS{i}' for i in range(1,7)]+['EPS1','FS1','FS2']+[f'TS{i}' for i in range(1,5)]+['VS1','CE','CP','SE']
  cols=[];names=[];rawhash={}
  for s in sensors:
   data=z.read(s+'.txt');rawhash[s]=hashlib.sha256(data).hexdigest();a=np.loadtxt(io.BytesIO(data));assert len(a)==2205 and np.isfinite(a).all()
   cols.extend([a.mean(1),a.std(1,ddof=0),a.min(1),a.max(1)]);names.extend([s+'_'+n for n in ['mean','std','min','max']]);del a,data
 x=np.column_stack(cols);y=(profile[:,2]==2).astype(int)
 _,g=np.unique(profile[:,[0,1,3]],axis=0,return_inverse=True)
 assert x.shape==(2205,68) and profile.shape==(2205,5) and set(y)=={0,1}
 np.savez_compressed(out/'features.npz',X=x,y=y,profile=profile,groups=g,feature_names=names)
 outer=StratifiedGroupKFold(5,shuffle=True,random_state=20261109)
 summaries=[]
 for fold,(dev,te) in enumerate(outer.split(x,y,g),1):
  tr0,va0=next(StratifiedGroupKFold(4,shuffle=True,random_state=20261109+fold).split(x[dev],y[dev],g[dev]));tr,va=dev[tr0],dev[va0]
  for ix in [tr,va,te]:assert len(np.unique(y[ix]))==2
  assert not(set(g[tr])&set(g[va]) or set(g[tr])&set(g[te]) or set(g[va])&set(g[te]))
  d=out/f'fold{fold}';d.mkdir();(d/'training_masks').mkdir()
  b=np.tile([.15,.20,.25,.30],17);delta=np.tile([.05,.10,.15,.20],17);pi=y[tr].mean();q0=b-pi*delta;q1=b+(1-pi)*delta
  assert q0.min()>=0 and q1.max()<=1
  inputdata={'train_rows':tr,'val_rows':va,'test_rows':te,'b':b,'delta':delta,'q0':q0,'q1':q1}
  xx={}
  for role,ix,ridx in [('train',tr,0),('val',va,1),('test',te,2)]:
   u=np.random.default_rng(52000+100*fold+ridx).random(x[ix].shape)
   q=b[None,:] if role=='test' else np.where(y[ix,None]==1,q1[None,:],q0[None,:])
   a=x[ix].copy();a[u<q]=np.nan;xx[role]=a
   inputdata.update({role+'_U':u,role+'_X':a,role+'_y':y[ix]})
  med=np.nanmedian(xx['train'],axis=0);med=np.where(np.isfinite(med),med,0)
  mean=np.nanmean(xx['train'],axis=0);sd=np.nanstd(xx['train'],axis=0);sd=np.where(sd>0,sd,1)
  assert np.isfinite(mean).all() and np.isfinite(sd).all()
  inputdata.update(medians=med,ma_mean=mean,ma_sd=sd)
  np.savez_compressed(d/'inputs.npz',**inputdata)
  for role in ['val','test']:
   ev=build_eval_inputs(xx[role]);np.savez_compressed(d/f'{role}_inputs.npz',**{k:v[0] for k,v in ev.items()})
  q,n,_=estimate_class_rates(xx['train'],y[tr])
  for a in [.25,.5,1]:
   expected=a*(n[:,None]*(q.max(0)[None,:]-q)).sum(0);br=class_blind_rates(q,n,a)
   assert np.allclose(expected,br*(~np.isnan(xx['train'])).sum(0),rtol=0,atol=1e-10)
   for seed in [19107,19108,19109]:
    u=np.random.default_rng(seed).random(xx['train'].shape)
    for arm in ['C','B']:
     _,info=mask_class_conditional(xx['train'],y[tr],a,uniforms=u) if arm=='C' else mask_class_blind(xx['train'],br,uniforms=u)
     mask=info['new_mask'];assert not(mask & np.isnan(xx['train'])).any()
     np.savez_compressed(d/'training_masks'/f'{arm}{int(a*100):03d}_s{seed}.npz',new_mask=mask,U=u,expected_col=expected)
  summaries.append({'fold':fold,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'train_positive':int(y[tr].sum()),'val_positive':int(y[va].sum()),'test_positive':int(y[te].sum()),'train_groups':len(set(g[tr])),'val_groups':len(set(g[va])),'test_groups':len(set(g[te])),'pi_train':float(pi),'group_intersections':0})
 dump(out/'data_summary.json',{'n':len(y),'positive':int(y.sum()),'group_count':len(np.unique(g)),'feature_count':68,'raw_zip_sha256':sha(zpath),'raw_sensor_sha256':rawhash,'folds':summaries})
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':prepare()
