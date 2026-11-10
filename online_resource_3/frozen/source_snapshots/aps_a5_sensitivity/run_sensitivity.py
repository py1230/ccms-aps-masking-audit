from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,ctypes,gzip,hashlib,json,platform,resource,shutil,subprocess,sys,time,traceback,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from aps_io import load_aps_csv,split_features
GROUPS=['S0','S1_10','S1_30','S2','S3']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def arrsha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+json.dumps(list(a.shape),separators=(',',':')).encode()+a.tobytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cfg():return json.loads((HERE/'config.json').read_text())
def env():return {'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'executable':sys.executable,'platform':platform.platform()}
def log(root,obj):
 obj={'timestamp':stamp(),**obj}
 with (root/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
 print(json.dumps(obj),flush=True)
def setup():
 old=HERE.parent/'aps_a5_20261108';o=json.loads((old/'config.json').read_text())
 assert not (HERE/'config.json').exists()
 c={'splits':[11,22,33,44,55],'references':[19107,19108,19109],'initials':['C','D'],'repeats':[1,2,3],'steps':[5,20,50],'dispersion_steps':100,'batch_size':200000,'expected_fits':270,'max_run_seconds':14400,'threads':2,'model_params':o['model_params'],'input_environment':o['input_environment'],'design':'fixed finite-step initial-state sensitivity; no performance-based selection; no CI'}
 dump(HERE/'config.json',c);sources=[]
 for dirname in ['raw_data','reference']:
  shutil.copytree(old/dirname,HERE/dirname)
  for f in (HERE/dirname).rglob('*'):
   if f.is_file():sources.append({'local':str(f.relative_to(HERE)),'source':str(old/f.relative_to(HERE)),'sha256':sha(f)})
 dump(HERE/'input_sources.json',sources)
 cmd=['cc','-O3','-shared','-fPIC',str(HERE/'source/switch.c'),'-o',str(HERE/'source/switch.dylib')];subprocess.run(cmd,check=True)
 dump(HERE/'compiler.json',{'command':cmd,'version':subprocess.check_output(['cc','--version'],text=True),'source_sha':sha(HERE/'source/switch.c'),'binary_sha':sha(HERE/'source/switch.dylib')});dump(HERE/'environment.json',env())
def load():
 d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');x,y=split_features(d);del d
 d,_=load_aps_csv(HERE/'raw_data/aps_failure_test_set.csv');xt,yt=split_features(d);del d
 assert x.shape==(60000,170) and xt.shape==(16000,170) and yt.sum()==375
 return x,y,xt,yt
def lib():
 l=ctypes.CDLL(str(HERE/'source/switch.dylib'));p=ctypes.c_void_p;l.a5_switch.argtypes=[p,p,p,p,p,p,p,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_uint32,p];l.a5_switch.restype=ctypes.c_size_t;return l

def chain(initial,ref,allowed,seed,steps,out):
 out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(seed);rng0=rng.bit_generator.state;t=time.perf_counter();l=lib();m=initial.astype(np.uint8);allowed=allowed.astype(np.uint8);rr,cc=np.nonzero(initial);rr=rr.astype(np.uint32);cc=cc.astype(np.uint32);e=len(rr);np.savez_compressed(out/'initial_edges.npz',rows=rr,cols=cc)
 refrow=ref.sum(1);refcol=ref.sum(0);batches=[];diags=[];offset=0;accepted=0
 with gzip.open(out/'accepted_trace.u32.gz','wb',compresslevel=3) as tracefile:
  for step in steps:
   while offset<step*e:
    n=min(cfg()['batch_size'],step*e-offset);before=rng.bit_generator.state;a=rng.integers(0,e,n,dtype=np.uint32);b=rng.integers(0,e,n,dtype=np.uint32);lazy=rng.integers(0,2,n,dtype=np.uint8);tr=np.empty((n,3),dtype=np.uint32)
    used=l.a5_switch(m.ctypes.data,allowed.ctypes.data,rr.ctypes.data,cc.ctypes.data,a.ctypes.data,b.ctypes.data,lazy.ctypes.data,n,m.shape[1],offset,tr.ctypes.data);tracefile.write(tr[:used].astype('<u4',copy=False).tobytes());accepted+=used
    batches.append({'start':offset,'n':n,'rng_before':before,'rng_after':rng.bit_generator.state,'a_sha':arrsha(a),'b_sha':arrsha(b),'lazy_sha':arrsha(lazy),'accepted':used});offset+=n
   mb=m.astype(bool);rowerr=int(np.max(abs(mb.sum(1).astype(int)-refrow)));colerr=int(np.max(abs(mb.sum(0).astype(int)-refcol)));forbid=int((mb&~allowed.astype(bool)).sum());assert rowerr==colerr==forbid==0
   np.savez_compressed(out/f'mask_{step}E.npz',new_mask=mb,row_deleted=mb.sum(1),col_deleted=mb.sum(0))
   diags.append({'step':step,'attempts':offset,'accepted':accepted,'acceptance_rate':accepted/offset,'row_error':rowerr,'col_error':colerr,'forbidden':forbid,'relocated_from_C':1-float((mb&ref).sum()/e),'relocated_from_initial':1-float((mb&initial).sum()/e),'mask_sha':arrsha(mb),'seconds_cumulative':time.perf_counter()-t})
 dump(out/'chain.json',{'seed':seed,'E':e,'steps':steps,'rng_initial':rng0,'rng_final':rng.bit_generator.state,'batches':batches,'diagnostics':diags,'seconds':time.perf_counter()-t,'initial_sha':arrsha(initial),'reference_sha':arrsha(ref)})
 return mb

def matrix(x,med):return np.hstack([np.where(np.isnan(x),med,x),np.isnan(x).astype(float)])
def smoke():
 root=HERE/'smoke';root.mkdir(exist_ok=False);x,y,xt,yt=load();pr=np.load(HERE/'reference/split11/preprocessing.npz');xx=x[pr['train_rows']];yy=y[pr['train_rows']];del x,y;ref=np.load(HERE/'reference/split11/training_masks/C025_seed19107.npz')['new_mask'];allowed=~np.isnan(xx)
 try:
  d=chain(ref,ref,allowed,99900001,[100],root/'dispersion');assert not np.array_equal(d,ref);m=chain(d,ref,allowed,99900002,[5,20,50],root/'continuation');xx[m]=np.nan;mat=matrix(xx,pr['medians']);t=time.perf_counter();model=HistGradientBoostingClassifier(**cfg()['model_params']).fit(mat,yy);secs=time.perf_counter()-t;del mat,xx
  joblib.dump(model,root/'interface_model.joblib',compress=3);p=model.predict_proba(matrix(xt[:300],pr['medians']))[:,1];p2=joblib.load(root/'interface_model.joblib').predict_proba(matrix(xt[:300],pr['medians']))[:,1];assert np.array_equal(p,p2)
  dump(root/'result.json',{'status':'passed','no_AP_computed':True,'n_train':42000,'fits':1,'fit_seconds':secs,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'reload_max_error':float(abs(p-p2).max()),'environment':env(),'timestamp':stamp()});print((root/'result.json').read_text(),flush=True)
 except BaseException:dump(root/'failure.json',{'traceback':traceback.format_exc(),'timestamp':stamp()});raise

def prepare_masks():
 root=HERE/'masks';root.mkdir(exist_ok=False);x,y,xt,yt=load();del xt,yt,y;start=time.perf_counter();c=cfg()
 try:
  for split in c['splits']:
   pr=np.load(HERE/f'reference/split{split}/preprocessing.npz');allowed=~np.isnan(x[pr['train_rows']])
   for rs in c['references']:
    ref=np.load(HERE/f'reference/split{split}/training_masks/C025_seed{rs}.npz')['new_mask'];base=root/f'split{split}'/f'ref{rs}';d=chain(ref,ref,allowed,920000000+split*100000+rs,[100],base/'initial_D');assert not np.array_equal(d,ref)
    for ii,initial in enumerate(c['initials']):
     startmask=ref if initial=='C' else d
     for rep in c['repeats']:
      seed=930000000+split*100000+rs*10+ii*10000000+rep;chain(startmask,ref,allowed,seed,c['steps'],base/initial/f'rep{rep}');log(root,{'event':'chain_complete','split':split,'ref':rs,'initial':initial,'repeat':rep,'elapsed_seconds':time.perf_counter()-start})
  dump(root/'complete.json',{'status':'complete','chains':90,'initial_D_chains':15,'masks':270,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'performance_accessed':False})
 except BaseException:dump(root/'failure.json',{'traceback':traceback.format_exc(),'timestamp':stamp()});raise

def freeze():
 assert not (HERE/'freeze.json').exists();assert json.loads((HERE/'masks/complete.json').read_text())['status']=='complete';assert json.loads((HERE/'smoke/result.json').read_text())['status']=='passed';assert len(list((HERE/'masks').glob('split*/ref*/[CD]/rep*/mask_*E.npz')))==270
 files={}
 for f in sorted(HERE.rglob('*')):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.log':continue
  if f.relative_to(HERE).parts[0] in ['run_results','verification']:continue
  files[str(f.relative_to(HERE))]=sha(f)
 dump(HERE/'freeze.json',{'timestamp':stamp(),'environment':env(),'files':files,'scope':'science code, protocol, configuration, copied references, all 270 masks and initialization traces before formal fitting'})
def check():
 for k,v in cfg()['input_environment'].items():assert env()[k]==v,(k,env()[k],v)
 for n,h in json.loads((HERE/'freeze.json').read_text())['files'].items():assert sha(HERE/n)==h,n

def run():
 check();root=HERE/'run_results';root.mkdir(exist_ok=False);x,y,xt,yt=load();c=cfg();evalz=np.load(HERE/'reference/test_evaluation_masks.npz');conditions=list(evalz.files);fits=0;start=time.perf_counter()
 try:
  for split in c['splits']:
   pr=np.load(HERE/f'reference/split{split}/preprocessing.npz');ids=pr['train_rows'];xx=x[ids];yy=y[ids];med=pr['medians'];calc=np.nan_to_num(np.nanmedian(xx,axis=0),nan=0.);assert np.array_equal(calc,med)
   for arm in ['C025','B025','BASE']:
    tags=['BASE'] if arm=='BASE' else [f'{arm}_seed{s}' for s in c['references']]
    for tag in tags:
     z=np.load(HERE/f'reference/split{split}/predictions/{tag}__test.npz');assert np.array_equal(z['y'],yt) and np.array_equal(z['row_indices'],np.arange(len(yt)))
     for cond in conditions:assert z[cond].shape==yt.shape and np.isfinite(z[cond]).all()
   for initial in c['initials']:
    for rep in c['repeats']:
     for step in c['steps']:
      dest=root/f'split{split}'/initial/f'rep{rep}'/f'{step}E';dest.mkdir(parents=True);records=[]
      for rs in c['references']:
       mp=HERE/f'masks/split{split}/ref{rs}/{initial}/rep{rep}/mask_{step}E.npz';m=np.load(mp)['new_mask'];xm=xx.copy();xm[m]=np.nan;train=matrix(xm,med);trainsha=arrsha(train);del xm;t=time.perf_counter();model=HistGradientBoostingClassifier(**c['model_params']).fit(train,yy);fitsec=time.perf_counter()-t;fits+=1;del train
       payload={'model':model,'medians':med,'train_rows':ids,'split':split,'initial':initial,'repeat':rep,'step':step,'reference_seed':rs,'model_seed':2026,'mask_sha':arrsha(m),'training_matrix_sha':trainsha,'training_labels_sha':arrsha(yy),'config_sha':sha(HERE/'config.json')};path=dest/f'model_ref{rs}.joblib';joblib.dump(payload,path,compress=3);preds={};t=time.perf_counter()
       for cond in conditions:
        xe=xt.copy();xe[evalz[cond]]=np.nan;preds[cond]=model.predict_proba(matrix(xe,med))[:,1];assert np.isfinite(preds[cond]).all();del xe
       predsec=time.perf_counter()-t;np.savez_compressed(dest/f'predictions_ref{rs}.npz',row_indices=np.arange(len(yt)),y=yt,**preds)
       rec={'reference_seed':rs,'fit_seconds':fitsec,'prediction_seconds':predsec,'n_iter':int(model.n_iter_),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_sha':sha(path),'mask_sha':arrsha(m),'training_matrix_sha':trainsha};records.append(rec);dump(dest/'fit_records.json',records);del model,payload,preds,m;gc.collect();log(root,{'event':'fit_complete','fits':fits,'split':split,'initial':initial,'rep':rep,'step':step,'reference':rs,'fit_seconds':fitsec,'peak_rss_bytes':rec['peak_rss_bytes'],'elapsed_seconds':time.perf_counter()-start})
       if time.perf_counter()-start>c['max_run_seconds']:raise TimeoutError('formal task exceeded frozen 14400 second ceiling')
  assert fits==270;dump(root/'complete.json',{'status':'complete','fits':fits,'seconds':time.perf_counter()-start,'environment':env(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'performance_computed_during_run':False});log(root,{'event':'complete','fits':fits})
 except BaseException:dump(root/'failure.json',{'traceback':traceback.format_exc(),'timestamp':stamp(),'fits_completed':fits});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','smoke','prepare_masks','freeze','check','run']);a=p.parse_args()
 with threadpool_limits(limits=2):globals()[a.action]()
if __name__=='__main__':main()
