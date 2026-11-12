"""Verify delivered SECOM draws from saved counts. No new resampling or training."""
from pathlib import Path
import argparse, hashlib, io, json, zipfile
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
ap.add_argument('--esm-zip',type=Path)
ap.add_argument('--out',type=Path)
a=ap.parse_args(); root=a.data_dir
archive=zipfile.ZipFile(a.esm_zip) if a.esm_zip else None
resource=root.parents[1]
prefix='historical_recovery/protocol_content_recovery_20261111/sources/'
def source(name):
 return archive.read('online_resource_3/'+prefix+name) if archive else (resource/prefix/name).read_bytes()
def digest(data):return hashlib.sha256(data).hexdigest()
def readj(path):return json.loads(path.read_text())
summary=readj(root/'exports/secom_boot20k_summary.json')
old=json.loads(source('secom_summary.json'))
counts=np.load(root/'exports/secom_boot20k_resample_counts.npz',allow_pickle=False)
draws=np.load(root/'exports/secom_boot20k_draws.npz',allow_pickle=False)

def weighted_ap_all(y,p,weights):
 # Aggregate tie blocks first, then compute precision at each score threshold.
 order=np.argsort(-p,kind='stable'); starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
 out=[]
 for start in range(0,len(weights),512):
  w=np.asarray(weights[start:start+512,order],dtype=np.float64)
  mass=np.add.reduceat(w,starts,axis=1)
  pos=np.add.reduceat(w*y[order],starts,axis=1)
  den=np.cumsum(mass,axis=1);tp=np.cumsum(pos,axis=1)
  precision=np.divide(tp,den,out=np.zeros_like(tp),where=den>0)
  out.append((pos*precision).sum(axis=1)/pos.sum(axis=1))
 return np.concatenate(out)

r={'identity':'independent deterministic replay of saved counts; no new bootstrap draws','manifest':[],'inputs':[],'fold_counts':[],'cells':{},'receipt_checks':{}}
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
 if not line.strip() or line.startswith('#'):continue
 h,name=line.split(None,1);name=name.strip().lstrip('*');p=root/name
 r['manifest'].append({'path':name,'pass':p.exists() and digest(p.read_bytes())==h})
for i in range(5):
 data=source(f'secom_fold{i}.npz');expected=next(v for k,v in summary['inputs'].items() if k.endswith(f'secom_fold{i}.npz'))
 r['inputs'].append({'fold':i,'sha256':digest(data),'match_declared':digest(data)==expected})
 z=np.load(io.BytesIO(data),allow_pickle=False);y=z['y_test'];w=counts[f'fold{i}']
 r['fold_counts'].append({'fold':i,'shape':list(w.shape),'dtype':str(w.dtype),'positive_count':int(y.sum()),'nonnegative_integer':bool(np.issubdtype(w.dtype,np.integer) and np.all(w>=0)),'row_counts_preserved':bool(np.all(w.sum(axis=1)==len(y))),'class_counts_preserved':bool(np.all(w[:,y==1].sum(axis=1)==sum(y==1)) and np.all(w[:,y==0].sum(axis=1)==sum(y==0)))})
 assert w.shape==(20000,len(y))
 for cell,ca,cb in [('fixed_C-B','C025','B025'),('selected_C-B','C_selected','B_selected')]:
  for fam,keys in summary['family'].items():
   key=f'{cell}__{fam}';stored=draws[key]
   assert stored.shape==(20000,5) and np.isfinite(stored).all()
   replay=[];points=[]
   for k in keys:
    pa=z[f'{ca}__test__{k}'];pb=z[f'{cb}__test__{k}']
    replay.append(weighted_ap_all(y,pa,w)-weighted_ap_all(y,pb,w))
    points.append(float(weighted_ap_all(y,pa,np.ones((1,len(y))))[0]-weighted_ap_all(y,pb,np.ones((1,len(y))))[0]))
   q=r['cells'].setdefault(key,{'folds':[]})
   q['folds'].append({'fold':i,'max_abs_draw_difference':float(np.max(np.abs(np.mean(replay,axis=0)-stored[:,i]))),'point_abs_difference':abs(float(np.mean(points))-summary['cells'][cell][fam]['per_fold'][i])})
   if i==4:
    dm=stored.mean(axis=1);s=summary['cells'][cell][fam];c=old['paired'][cell][fam]
    q.update({'delta_mean':s['delta_mean'],'ci95_from_draws':np.quantile(dm,[.025,.975]).tolist(),'ci_max_abs_difference':float(np.max(np.abs(np.quantile(dm,[.025,.975])-s['ci95']))),'first1000_ci_max_abs_difference':float(np.max(np.abs(np.quantile(dm[:1000],[.025,.975])-c['ci95']))),'ci_includes_zero':bool(s['ci95'][0]<0<s['ci95'][1])})
receipt=readj(root/'logs/execution_receipt.json');env=readj(root/'exports/secom_boot20k_environment.json')
for key,path in [('script_sha256','scripts/run_secom_boot20k.py'),('wrapper_sha256','scripts/run_with_receipt.sh'),('log_sha256','logs/run.log')]:r['receipt_checks'][key]=digest((root/path).read_bytes())==receipt[key]
r['receipt_checks']['exit_code_zero']=receipt['exit_code']==0
r['receipt_checks']['environment_script_sha256']=env['script_sha256']==digest((root/'scripts/run_secom_boot20k.py').read_bytes())
r['max_abs_draw_difference']=max(f['max_abs_draw_difference'] for c in r['cells'].values() for f in c['folds'])
r['all_pass']=bool(all(x['pass'] for x in r['manifest']) and all(x['match_declared'] for x in r['inputs']) and all(x['nonnegative_integer'] and x['row_counts_preserved'] and x['class_counts_preserved'] for x in r['fold_counts']) and all(r['receipt_checks'].values()) and r['max_abs_draw_difference']<1e-12 and all(c['ci_max_abs_difference']<1e-12 and c['first1000_ci_max_abs_difference']<1e-12 and all(f['point_abs_difference']<1e-12 for f in c['folds']) for c in r['cells'].values()))
if a.out:a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps({'all_pass':r['all_pass'],'manifest_count':len(r['manifest']),'max_abs_draw_difference':r['max_abs_draw_difference'],'cells':{k:{'point':v['delta_mean'],'ci95':v['ci95_from_draws']} for k,v in r['cells'].items()}},ensure_ascii=False))
assert r['all_pass']
