import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import argparse,json,hashlib,platform,sys,time,traceback,resource
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,sklearn,scipy,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits,threadpool_info
HERE=Path(__file__).resolve().parent;EXT=HERE.parent/'external_baseline_20261109';sys.path.insert(0,str(EXT))
from ma_tree import MADTClassifier
SEL=['S0','S1_10_101','S1_10_202','S1_10_303']
HP={'max_iter':100,'learning_rate':.1,'max_leaf_nodes':31,'min_samples_leaf':20,'max_bins':255,'l2_regularization':0.,'random_state':2026,'early_stopping':False}
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def arrload(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def env():return {'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'threads':threadpool_info()}
def specs():
 out=[(x,x,0,None) for x in ['M2','M3','M4']]
 for a in [.25,.5,1]:
  for arm in ['C','B']:
   for s in [19107,19108,19109]:out.append((f'{arm}{int(a*100):03d}_s{s}',arm,a,s))
 for depth in [3,5,7]:
  for a in [0,.001,.01,.1,1,10]:out.append((f'MA_d{depth}_a{a:g}','MA',a,depth))
 return out
SPECS=specs()
def mat(x,kind,inp):
 if kind=='M4':return x
 imp=np.where(np.isnan(x),inp['medians'][None,:],x)
 if kind=='MA':return (imp-inp['ma_mean'][None,:])/inp['ma_sd'][None,:]
 if kind=='M2':return imp
 return np.hstack([imp,np.isnan(x).astype(float)])
def choose(metrics):
 selection={};candidates={}
 for arm in ['C','B']:
  rows=[]
  for a in [0,.25,.5,1]:
   tag='M3' if a==0 else f'{arm}{int(a*100):03d}';mm=metrics[tag]
   rows.append({'config':tag,'alpha':a,'s0':mm['S0'],'worst4':min(mm.values()),'eligible':mm['S0']>=metrics['M3']['S0']-.01})
  selection[arm]=max([r for r in rows if r['eligible']],key=lambda r:(r['worst4'],-r['alpha']));candidates[arm]=rows
 rows=[]
 for tag,arm,a,depth in SPECS:
  if arm!='MA':continue
  mm=metrics[tag];rows.append({'config':tag,'alpha':a,'depth':depth,'s0':mm['S0'],'worst4':min(mm[k] for k in SEL),'auc':mm['auc'],'rho':mm['rho']})
 zero=max([r for r in rows if r['alpha']==0],key=lambda r:(r['worst4'],-r['depth']))
 selection['MA0']=zero
 for r in rows:r['eligible']=r['s0']>=zero['s0']-.01
 selection['MA_common']=max([r for r in rows if r['eligible']],key=lambda r:(r['worst4'],-r['depth'],-r['alpha']))
 bestauc=max(r['auc'] for r in rows)
 selection['MA_tradeoff']=min([r for r in rows if r['auc']>=.95*bestauc],key=lambda r:(r['rho'],-r['auc'],r['depth'],r['alpha']))
 candidates['MA']=rows
 return {'chosen':selection,'candidates':candidates,'timestamp_utc':stamp(),'test_predictions_created':False}
def freeze():
 assert not(HERE/'run_results').exists()
 files=list((HERE/'prepared').rglob('*'))+list((HERE/'source').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'prepare.py',HERE/'run.py',HERE/'audit.py',HERE/'execution_schedule.json',EXT/'ma_tree.py',EXT/'extract_ma_tree.py',EXT/'extraction_manifest.json',EXT/'vendor/malearn/LICENSE.md',HERE/'data/uci447_legacy.zip']+list((EXT/'vendor/malearn/malearn/estimators').glob('*.py'))
 files=[p for p in files if p.is_file() and '__pycache__' not in str(p)]
 dump(HERE/'freeze.json',{'timestamp_utc':stamp(),'environment':env(),'hgb_params':HP,'expected_fits':len(SPECS)*len(json.loads((HERE/'execution_schedule.json').read_text())['folds']),'files':{str(p.relative_to(HERE.parent)):sha(p) for p in files}})
 print('FROZEN',len(files),'files',len(SPECS)*len(json.loads((HERE/'execution_schedule.json').read_text())['folds']),'fits',flush=True)
def run():
 fr=json.loads((HERE/'freeze.json').read_text())
 for key in ['python','numpy','scipy','sklearn','joblib']:
  assert env()[key]==fr['environment'][key],f'Environment mismatch: {key}'
 for name,h in fr['files'].items():assert sha(HERE.parent/name)==h,name
 out=HERE/'run_results';out.mkdir(exist_ok=False);start=time.perf_counter();count=0
 dump(out/'environment.json',{'started_utc':stamp(),'freeze_sha256':sha(HERE/'freeze.json'),**env(),'model_params':HP})
 def log(**obj):
  rec={'time_utc':stamp(),'elapsed_seconds':time.perf_counter()-start,**obj}
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
  print(json.dumps(rec,ensure_ascii=False),flush=True)
 try:
  for fold in json.loads((HERE/'execution_schedule.json').read_text())['folds']:
   d=out/f'fold{fold}';d.mkdir();(d/'models').mkdir();(d/'predictions').mkdir()
   inp=arrload(HERE/f'prepared/fold{fold}/inputs.npz');val=arrload(HERE/f'prepared/fold{fold}/val_inputs.npz');val={k:val[k] for k in SEL}
   preds={};fitrecords=[]
   for tag,arm,a,s in SPECS:
    xx=inp['train_X'].copy()
    if arm in ['C','B']:xx[arrload(HERE/f'prepared/fold{fold}/training_masks/{tag}.npz')['new_mask']]=np.nan
    if arm=='MA':model=MADTClassifier(alpha=a,max_depth=s,criterion='gini',random_state=2026)
    else:model=HistGradientBoostingClassifier(**HP)
    t=time.perf_counter()
    if arm=='MA':model.fit(mat(xx,arm,inp),inp['train_y'],M=np.isnan(xx))
    else:model.fit(mat(xx,arm,inp),inp['train_y'])
    seconds=time.perf_counter()-t;count+=1
    pp=d/'models'/f'{tag}.joblib';joblib.dump({'model':model,'arm':arm,'alpha':a,'seed_or_depth':s,'tag':tag,'train_rows':inp['train_rows']},pp,compress=3)
    pred={k:model.predict_proba(mat(x,arm,inp))[:,1] for k,x in val.items()}
    for v in pred.values():assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
    np.savez_compressed(d/'predictions'/f'{tag}_val.npz',y=inp['val_y'],rows=inp['val_rows'],**pred);preds[tag]=pred
    rho=float(model.compute_missingness_reliance(mat(val['S0'],arm,inp),np.isnan(val['S0']))) if arm=='MA' else None
    fitrecords.append({'tag':tag,'arm':arm,'fit_seconds':seconds,'model_sha256':sha(pp),'rho':rho,'n_iter':int(model.n_iter_) if arm!='MA' else None})
    log(event='fit',fold=fold,tag=tag,fits_completed=count,fit_seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
   dump(d/'fit_records.json',fitrecords)
   ens={};metrics={}
   for config in list(dict.fromkeys(t.split('_s')[0] if arm in ['C','B'] else t for t,arm,_,_ in SPECS)):
    tags=[t for t,arm,_,_ in SPECS if (t.split('_s')[0] if arm in ['C','B'] else t)==config]
    ens[config]={k:np.mean([preds[t][k] for t in tags],axis=0) for k in SEL}
    metrics[config]={k:float(average_precision_score(inp['val_y'],v)) for k,v in ens[config].items()}
    if config.startswith('MA_'):
     metrics[config]['auc']=float(roc_auc_score(inp['val_y'],ens[config]['S0']));metrics[config]['rho']=next(r['rho'] for r in fitrecords if r['tag']==config)
   selection=choose(metrics);dump(d/'selection.json',selection);dump(d/'validation_metrics.json',metrics)
   log(event='selection_frozen',fold=fold,selection_sha256=sha(d/'selection.json'))
   test=arrload(HERE/f'prepared/fold{fold}/test_inputs.npz');testpred={}
   for tag,arm,a,s in SPECS:
    model=joblib.load(d/'models'/f'{tag}.joblib')['model'];pp={k:model.predict_proba(mat(x,arm,inp))[:,1] for k,x in test.items()};testpred[tag]=pp
    np.savez_compressed(d/'predictions'/f'{tag}_test.npz',y=inp['test_y'],rows=inp['test_rows'],**pp)
   ensemble={'val_y':inp['val_y'],'test_y':inp['test_y'],'val_rows':inp['val_rows'],'test_rows':inp['test_rows']}
   for config,pp in ens.items():
    for k,v in pp.items():ensemble[f'{config}__val__{k}']=v
    tags=[t for t,arm,_,_ in SPECS if (t.split('_s')[0] if arm in ['C','B'] else t)==config]
    for k in test:ensemble[f'{config}__test__{k}']=np.mean([testpred[t][k] for t in tags],axis=0)
   np.savez_compressed(d/'ensemble_predictions.npz',**ensemble)
   log(event='fold_complete',fold=fold,fits_completed=count)
  dump(out/'completion.json',{'status':'completed','fits':count,'finished_utc':stamp(),'elapsed_seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
  log(event='complete',fits=count)
 except Exception:
  dump(out/'failure.json',{'timestamp_utc':stamp(),'fits_completed':count,'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 with threadpool_limits(limits=2):
  arg=sys.argv[1]
  if arg=='freeze':freeze()
  elif arg=='run':run()
