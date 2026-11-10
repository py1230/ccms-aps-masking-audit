"""Read-only historical prediction anchor; output only audit JSON/stdout, no historical mutation."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,platform,time,ast
from pathlib import Path
import numpy as np,sklearn
from sklearn.metrics import average_precision_score as AP
from threadpoolctl import threadpool_limits
threadpool_limits(2)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PKG=Path('/Users/bgc/Documents/ChatGPT/research/rr/02_原完整复现包/CCMS论文修订完整资料包_2026-11-01_192149')
SRC=PKG/'03_原始项目归档/CCMS核查_2026-10-31'
ENTRY=PKG/'02_本轮代码与结果/scripts/run_supplement.py'
sys.path[:0]=[str(SRC/'src'),str(SRC/'scripts')]
from sim_nuisance import gen_data,base_rates
from pipeline import MedianImputer,_make_model
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
from run_cb_firstlook import stress_unit
ALPHAS=[.25,.5,1.]
# Execute only source function definitions; archived top-level SOURCE path is wrong after relocation.
nodes=[n for n in ast.parse(ENTRY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['matrix','fit_predict','metrics','select']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ENTRY),'exec'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((ROOT/'data/simulation/simulation_continuous.json').read_text())
result={'purpose':'Historical anchor only; separate from new formal seeds; consistency does not establish historical origin.', 'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'historical_environment':old['environment'],'source_hashes':{str(p):sha(p) for p in [Path(__file__),ENTRY,SRC/'scripts/sim_nuisance.py',SRC/'src/ccms.py',SRC/'src/pipeline.py',SRC/'scripts/run_cb_firstlook.py']},'seeds':[],'fits_completed':0}
for ds in range(501,506):
 t=time.time(); X,y,meta=gen_data(ds); rng=np.random.default_rng(ds+7);parts=[[],[],[]]
 for c in [0,1]:
  idx=rng.permutation(np.flatnonzero(y==c));i,j=round(.6*len(idx)),round(.8*len(idx))
  for k,z in enumerate([idx[:i],idx[i:j],idx[j:]]):parts[k].extend(z)
 tr,va,te=[np.sort(v) for v in parts]
 archive=np.load(ROOT/f'data/simulation/simulation_seed{ds}.npz');historical=next(v for v in old['seeds'] if v['data_seed']==ds)
 index_match={k:bool(np.array_equal(archive[k],v)) for k,v in [('train_rows',tr),('val_rows',va),('test_rows',te),('y_val',y[va]),('y_test',y[te])]}
 b,delta=base_rates(ds);b[:6]=.4
 def missing(xx,yy,mult,seed):
  pi=yy.mean();q1=b+(1-pi)*delta*mult;q0=b-pi*delta*mult;U=np.random.default_rng(seed).random(xx.shape);z=xx.copy();z[U<np.where(yy[:,None]==1,q1,q0)]=np.nan
  return z,{'q1':q1.tolist(),'q0':q0.tolist(),'expected_overall_missing':float((pi*q1+(1-pi)*q0).mean()),'actual_missing':float(np.isnan(z).mean())}
 x,mt=missing(X[tr],y[tr],1,ds*10+1);v,mv=missing(X[va],y[va],1,ds*10+2)
 ve={'S0':v};ve.update({f'S1_10_{s}':stress_unit(v,.1,s)[0] for s in [101,202,303]})
 ee={};target_meta={}
 for m in old['target_gap_multipliers']:ee[f'gap_{m}'],target_meta[f'gap_{m}']=missing(X[te],y[te],m,ds*10+3)
 ev={f'val__{k}':z for k,z in ve.items()};ev.update({f'test__{k}':z for k,z in ee.items()})
 imp=MedianImputer().fit(x);ps={bb:fit_predict(x,y[tr],imp,ev,bb) for bb in ['M3','M4']};result['fits_completed']+=2
 q,cnt,_=estimate_class_rates(x,y[tr])
 for a in ALPHAS:
  bv=class_blind_rates(q,cnt,a)
  for arm in ['C','B']:
   runs=[]
   for ms in [9107,9108,9109]:
    xm,info=mask_class_conditional(x,y[tr],a,seed=ms) if arm=='C' else mask_class_blind(x,bv,seed=ms)
    runs.append(fit_predict(xm,y[tr],imp,ev));result['fits_completed']+=1
   ps[f'{arm}{round(a*100):03d}']={k:np.mean([z[k] for z in runs],0) for k in ev}
 sel=select(y[va],{c:{k[5:]:p for k,p in z.items() if k.startswith('val__')} for c,z in ps.items() if c!='M4'})
 for arm in ['C','B']:ps[arm+'_selected']=ps[sel[arm]['chosen']['config']]
 errors={c+'__'+k:float(np.max(np.abs(p-archive[c+'__'+k]))) for c,z in ps.items() for k,p in z.items()}
 ap_err={c+'__'+k:abs(float(AP(y[te],p))-historical['metrics'][c][k[6:]]) for c,z in ps.items() for k,p in z.items() if k.startswith('test__')}
 rec={'data_seed':ds,'generated_X_bytes_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'metadata':meta,'index_label_match':index_match,'missing_metadata_exact':mt==historical['train_missing'] and mv==historical['validation_missing'] and target_meta==historical['target_missing'],'prediction_arrays_checked':len(errors),'max_abs_prediction_error':max(errors.values()),'prediction_array_errors':errors,'target_ap_checked':len(ap_err),'max_abs_target_ap_error':max(ap_err.values()),'selection_exact':sel==historical['source_selection'],'seconds':time.time()-t}
 result['seeds'].append(rec)
 (OUT/'historical_anchor_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in rec.items() if k!='prediction_array_errors'},ensure_ascii=False),flush=True)
result['status']='passed' if all(all(s['index_label_match'].values()) and s['missing_metadata_exact'] and s['max_abs_prediction_error']==0 and s['max_abs_target_ap_error']<1e-12 and s['selection_exact'] for s in result['seeds']) else 'failed'
(OUT/'historical_anchor_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(result['status'],result['fits_completed'],flush=True)
