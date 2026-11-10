"""Independent checks: no imports from run.py, ccms.py or pressure builders."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,hashlib,time,zipfile,io
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'external_baseline_20261109'))
from ma_tree import MADTClassifier
SEL=['S0','S1_10_101','S1_10_202','S1_10_303']
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def ap(y,p):
 order=np.argsort(-p,kind='stable');yy=y[order];ss=p[order];ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];tp=np.cumsum(yy)[ends];return float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/y.sum())
def auc(y,p):
 order=np.argsort(p,kind='stable');ss=p[order];yy=y[order];ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];start=0;s=0.
 for end in ends:
  s+=yy[start:end+1].sum()*(start+end+2)/2;start=end+1
 np_=y.sum();nn=len(y)-np_;return float((s-np_*(np_+1)/2)/(np_*nn))
def pressures(x):
 d={'S0':x}
 for seed in [101,202,303]:
  u=np.random.default_rng(seed).random(x.shape)
  for rate,name in [(.1,'S1_10'),(.3,'S1_30')]:
   m=x.copy();m[u<rate]=np.nan;d[f'{name}_{seed}']=m
 for seed in [1101,1202,1303]:
  r=np.random.default_rng(seed);ix=r.choice(len(x),int(round(.5*len(x))),replace=False);u=r.random((len(ix),x.shape[1]));m=x.copy();sub=m[ix];sub[u<.6]=np.nan;m[ix]=sub;d[f'S2_{seed}']=m
 for seed in [2101,2202,2303]:
  r=np.random.default_rng(seed);j=r.choice(x.shape[1],max(1,int(round(.3*x.shape[1]))),replace=False);m=x.copy();m[:,j]=np.nan;d[f'S3_{seed}']=m
 return d
def transform(x,arm,z):
 if arm=='M4':return x
 im=x.copy();ix=np.where(np.isnan(im));im[ix]=z['medians'][ix[1]]
 if arm=='M2':return im
 if arm=='MA':return (im-z['ma_mean'])/z['ma_sd']
 return np.column_stack((im,np.isnan(x).astype(float)))
def main():
 fr=json.loads((HERE/'freeze.json').read_text())
 for name,h in fr['files'].items():assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==h,name
 completed=json.loads((HERE/'run_results/completion.json').read_text());assert completed['status']=='completed' and completed['fits']==195
 expected={'M2','M3','M4'}|{f'{arm}{a:03d}_s{seed}' for arm in ['C','B'] for a in [25,50,100] for seed in [19107,19108,19109]}|{f'MA_d{depth}_a{a:g}' for depth in [3,5,7] for a in [0,.001,.01,.1,1,10]}
 assert len(expected)==39
 start=time.perf_counter();raw=load(HERE/'prepared/features.npz')
 with zipfile.ZipFile(HERE/'data/uci447_legacy.zip') as zraw:
  profile=np.loadtxt(zraw.open('profile.txt'),dtype=int);assert np.array_equal(profile,raw['profile']);assert np.array_equal((profile[:,2]==2).astype(int),raw['y'])
  sensors=[f'PS{i}' for i in range(1,7)]+['EPS1','FS1','FS2']+[f'TS{i}' for i in range(1,5)]+['VS1','CE','CP','SE']
  for j,sensor in enumerate(sensors):
   a=np.loadtxt(zraw.open(sensor+'.txt'));v=np.column_stack((a.mean(1),a.std(1,ddof=0),a.min(1),a.max(1)));assert np.array_equal(v,raw['X'][:,4*j:4*j+4]);del a,v
 seen=[];checks=0;maxerr=0.;maxap=0.;summaries=[];counts={'models':0,'probability_vectors':0,'ensemble_vectors':0,'training_masks':0,'evaluation_inputs':0}
 for fold in range(1,6):
  prep=HERE/f'prepared/fold{fold}';out=HERE/f'run_results/fold{fold}';z=load(prep/'inputs.npz')
  assert {p.stem for p in (out/'models').glob('*.joblib')}==expected
  assert {p.name for p in (out/'predictions').glob('*.npz')}=={tag+'_'+role+'.npz' for tag in expected for role in ['val','test']}
  ixs=[z[k+'_rows'] for k in ['train','val','test']]
  assert np.array_equal(np.sort(np.concatenate(ixs)),np.arange(len(raw['y'])))
  for a in range(3):
   for b in range(a):assert not(set(raw['groups'][ixs[a]])&set(raw['groups'][ixs[b]]));checks+=1
  seen.extend(ixs[2].tolist());p=z['train_y'].mean()
  for role,idx,ridx in [('train',ixs[0],0),('val',ixs[1],1),('test',ixs[2],2)]:
   assert np.array_equal(z[role+'_y'],raw['y'][idx]);base=np.tile([.15,.20,.25,.30],17);delta=np.tile([.05,.10,.15,.20],17)
   prob=base[None,:] if role=='test' else base[None,:]+(z[role+'_y'][:,None]-p)*delta[None,:]
   u=np.random.default_rng(52000+100*fold+ridx).random(z[role+'_X'].shape)
   assert np.array_equal(u,z[role+'_U']);m=raw['X'][idx].copy();m[u<prob]=np.nan
   assert np.array_equal(m,z[role+'_X'],equal_nan=True);checks+=3
  assert np.array_equal(np.nanmedian(z['train_X'],axis=0),z['medians']);checks+=1
  assert np.array_equal(np.nanmean(z['train_X'],axis=0),z['ma_mean']);checks+=1
  ss=np.nanstd(z['train_X'],axis=0);ss[ss==0]=1;assert np.array_equal(ss,z['ma_sd']);checks+=1
  orig=np.isnan(z['train_X']);yc=z['train_y'];q=np.stack([orig[yc==c].mean(0) for c in [0,1]]);qm=q.max(0);qbar=orig.mean(0);n=np.bincount(yc)
  for file in (prep/'training_masks').glob('*.npz'):
   zz=load(file);arm=file.stem[0];alpha=int(file.stem[1:4])/100;seed=int(file.stem.split('_s')[1]);u=np.random.default_rng(seed).random(orig.shape)
   a=alpha*(qm-q)/(1-q) if arm=='C' else alpha*(qm-qbar)/(1-qbar)
   prob=a[yc] if arm=='C' else a[None,:];mask=(~orig)&(u<prob)
   assert np.array_equal(u,zz['U']) and np.array_equal(mask,zz['new_mask']);counts['training_masks']+=1
   exp=alpha*np.sum(n[:,None]*(qm[None,:]-q),axis=0);assert np.max(np.abs(exp-zz['expected_col']))<1e-10;checks+=3
  inputs={}
  for role in ['val','test']:
   ev=load(prep/f'{role}_inputs.npz');re=pressures(z[role+'_X'])
   for k in ev:assert np.array_equal(ev[k],re[k],equal_nan=True);counts['evaluation_inputs']+=1
   inputs[role]=ev if role=='test' else {k:ev[k] for k in SEL}
  ensemble=load(out/'ensemble_predictions.npz');allpred={'val':{},'test':{}};metrics={};rho={};spec=[]
  for modelpath in sorted((out/'models').glob('*.joblib')):
   obj=joblib.load(modelpath);model=obj['model'];tag=obj['tag'];arm=obj['arm'];spec.append((tag,arm,obj['alpha'],obj['seed_or_depth']))
   assert np.array_equal(obj['train_rows'],z['train_rows']);checks+=1;counts['models']+=1
   params=model.get_params()
   if arm=='MA':
    assert params['alpha']==obj['alpha'] and params['max_depth']==obj['seed_or_depth'] and params['random_state']==2026 and params['criterion']=='gini'
   else:
    for key,value in fr['hgb_params'].items():assert params[key]==value,(tag,key)
    assert model.n_iter_==100 and not model.do_early_stopping_
   assert model.n_features_in_==(136 if arm in ['M3','C','B'] else 68);checks+=1
   for role in ['val','test']:
    pp=load(out/'predictions'/f'{tag}_{role}.npz');allpred[role][tag]=pp
    assert np.array_equal(pp['y'],z[role+'_y']) and np.array_equal(pp['rows'],z[role+'_rows']);checks+=2
    for k,x in inputs[role].items():
     p1=model.predict_proba(transform(x,arm,z))[:,1];err=float(np.max(np.abs(p1-pp[k])));maxerr=max(err,maxerr);assert err<1e-12;counts['probability_vectors']+=1
     ae=abs(ap(pp['y'],pp[k])-average_precision_score(pp['y'],pp[k]));assert ae<1e-12;maxap=max(maxap,ae)
   if arm=='MA':
    x=transform(inputs['val']['S0'],arm,z);mask=np.isnan(inputs['val']['S0']);rel=[]
    for row,m in zip(x,mask):
     node=model.root_;use=False
     while node.left_subtree is not None:
      use=use or bool(m[node.feature]);node=node.left_subtree if row[node.feature]<=node.threshold else node.right_subtree
     rel.append(use)
    rho[tag]=float(np.mean(rel))
  configs=sorted(set(t.split('_s')[0] if arm in ['C','B'] else t for t,arm,_,_ in spec))
  testmetrics={}
  for config in configs:
   tags=[t for t,arm,_,_ in spec if (t.split('_s')[0] if arm in ['C','B'] else t)==config]
   for role in ['val','test']:
    mm={}
    for k in inputs[role]:
     pred=np.mean([allpred[role][t][k] for t in tags],axis=0);saved=ensemble[f'{config}__{role}__{k}'];err=float(np.max(np.abs(pred-saved)));assert err<1e-12;maxerr=max(maxerr,err);counts['ensemble_vectors']+=1;mm[k]=ap(z[role+'_y'],saved)
    if role=='val':metrics[config]=mm
    else:testmetrics[config]={c:mm['S0'] if c=='S0' else float(np.mean([v for k,v in mm.items() if k.startswith(c+'_')])) for c in ['S0','S1_10','S1_30','S2','S3']}
  selected=json.loads((out/'selection.json').read_text());ind={}
  for arm in ['C','B']:
   options=[(a,'M3' if a==0 else f'{arm}{int(a*100):03d}') for a in [0,.25,.5,1]];options=[(a,k) for a,k in options if metrics[k]['S0']>=metrics['M3']['S0']-.01]
   ind[arm]=sorted(options,key=lambda z:(-min(metrics[z[1]].values()),z[0]))[0][1]
  ma=[(t,a,depth) for t,arm,a,depth in spec if arm=='MA'];zero=sorted([r for r in ma if r[1]==0],key=lambda r:(-min(metrics[r[0]].values()),r[2]))[0]
  ind['MA0']=zero[0]
  eligible=[r for r in ma if metrics[r[0]]['S0']>=metrics[zero[0]]['S0']-.01]
  ind['MA_common']=sorted(eligible,key=lambda r:(-min(metrics[r[0]].values()),r[2],r[1]))[0][0]
  aucs={r[0]:auc(z['val_y'],ensemble[f'{r[0]}__val__S0']) for r in ma};maxauc=max(aucs.values());eligible=[r for r in ma if aucs[r[0]]>=.95*maxauc]
  ind['MA_tradeoff']=sorted(eligible,key=lambda r:(rho[r[0]],-aucs[r[0]],r[2],r[1]))[0][0]
  for arm,k in ind.items():assert selected['chosen'][arm]['config']==k;checks+=1
  events=[json.loads(line) for line in (HERE/'run_results/events.jsonl').read_text().splitlines()];ev=[e for e in events if e.get('fold')==fold];sel_event=next(e for e in ev if e['event']=='selection_frozen');assert sel_event['selection_sha256']==hashlib.sha256((out/'selection.json').read_bytes()).hexdigest();checks+=1
  # Preserve every model result; all selected aliases point to actually saved configurations.
  for arm,k in ind.items():testmetrics[arm+'_selected' if arm in ['C','B'] else arm]=testmetrics[k]
  summaries.append({'fold':fold,'chosen':{k:v['config'] for k,v in selected['chosen'].items()},'ap':testmetrics})
  print('AUDITED fold',fold,'models',counts['models'],flush=True)
 assert sorted(seen)==list(range(len(raw['y'])));checks+=1
 assert counts['models']==195 and counts['probability_vectors']==3315 and counts['ensemble_vectors']==2295 and counts['training_masks']==90 and counts['evaluation_inputs']==130
 arms=['M2','M3','M4','C025','B025','C_selected','B_selected','MA0','MA_common','MA_tradeoff'];conds=['S0','S1_10','S1_30','S2','S3']
 mean={a:{c:float(np.mean([f['ap'][a][c] for f in summaries])) for c in conds} for a in arms}
 contrasts={}
 for left,right in [('C025','B025'),('C_selected','B_selected'),('MA_common','MA0'),('C_selected','MA_common'),('B_selected','MA_common'),('MA_tradeoff','MA0')]:
  contrasts[left+'-'+right]={}
  for c in conds:
   d=[f['ap'][left][c]-f['ap'][right][c] for f in summaries];contrasts[left+'-'+right][c]={'mean':float(np.mean(d)),'min':float(min(d)),'max':float(max(d)),'fold_differences':d}
 summary={'status':'passed','checks':checks,'counts':counts,'max_prediction_error':maxerr,'max_ap_error':maxap,'audit_seconds':time.perf_counter()-start,'folds':summaries,'mean_ap':mean,'contrasts':contrasts,'inference':'Five test folds share overlapping training samples. Means/ranges descriptive; no independent-fold significance claims.'}
 (HERE/'audit_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('ALL PASSED',counts,'maxerr',maxerr,'maxAP',maxap,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
