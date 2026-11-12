"""Read-only checks of recovered archives and saved point estimates; no resampling."""
from pathlib import Path
import ast, hashlib, json, zipfile, io, difflib
import numpy as np

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent.parent
TARGETS={
 'CCMS_v2.2_复核脚本与结果.zip':'a99cdec7b7582b5d9464ab57b927e88253de4869d8d4f1a16634c5129a16a4e2',
 'CCMS_Online_Resource_3_v2.1_路径修订版.zip':'a207d8bc98f11320ad9bff5fdefc2347e90e6af927bd27fc16d9bcc826637f0d',
}
def sha(data):return hashlib.sha256(data).hexdigest()
def ap(y,s):
 order=np.argsort(-s,kind='stable'); yy=y[order];ss=s[order]
 ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(y)-1]
 tp=np.cumsum(yy)[ends]
 return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/sum(y))
def differences(a,b,path=''):
 if type(a)!=type(b):return [path]
 if isinstance(a,dict):return sum((differences(a.get(k),b.get(k),path+'/'+k) for k in a.keys()|b.keys()),[])
 if isinstance(a,list):
  if len(a)!=len(b):return [path]
  return sum((differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
 return [] if a==b else [path]
def check():
 archives=[];total=0
 for name,target in TARGETS.items():
  p=ROOT/'original_zips'/name;assert sha(p.read_bytes())==target
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   manifest='SHA256SUMS.txt' if 'v2.2' in name else 'SHA256SUMS_Online_Resource_3.txt'
   count=0
   for line in z.read(manifest).decode().splitlines():
    if not line.strip():continue
    h,n=line.split('  ',1);assert sha(z.read(n))==h;count+=1
   assert count==len(z.namelist())-1
   if 'v2.2' in name:
    for n in z.namelist():assert z.read(n)==(ROOT/'v22_sources'/n).read_bytes()
   archives.append({'file':name,'sha256':target,'bytes':p.stat().st_size,'members':len(z.namelist()),'manifest_matches':count});total+=count
 result=json.loads((ROOT/'v22_sources/secom_bootstrap_20k_20261108.json').read_text())
 assert result['protocol']['bootstrap_replicates']==20000 and result['protocol']['random_seed']==20261108
 inputs=ROOT.parent/'protocol_content_recovery_20261111/sources';folds=[]
 for source in result['source_files']:
  p=inputs/Path(source['path']).name;assert len(p.read_bytes())==source['n_bytes'] and sha(p.read_bytes())==source['sha256']
  with np.load(p,allow_pickle=False) as f:folds.append({k:f[k] for k in f.files})
 assert len(folds)==5
 test=[]
 for f in folds:
  s=[set(f[k].tolist()) for k in ['train_rows','val_rows','test_rows']]
  assert not(s[0]&s[1] or s[0]&s[2] or s[1]&s[2]);test.extend(f['test_rows'].tolist())
 assert sorted(test)==list(range(1567))
 assert result['fold_partition']['fold_test_sizes']==[len(f['test_rows']) for f in folds]
 assert result['fold_partition']['fold_positive_counts']==[int(sum(f['y_test'])) for f in folds]
 errors=[];mean_checks=[]
 for condition,repeats in {'S0':['S0'],'S1_30':['S1_30_101','S1_30_202','S1_30_303']}.items():
  for pair,(arm_a,arm_b) in {'fixed_alpha_0.25':('C025','B025'),'validation_selected':('C_selected','B_selected')}.items():
   vals=[float(np.mean([ap(f['y_test'],f[f'{arm_a}__test__{c}']) for c in repeats])-np.mean([ap(f['y_test'],f[f'{arm_b}__test__{c}']) for c in repeats])) for f in folds]
   expected=result['point_estimates'][condition][pair];observed=np.r_[vals,np.mean(vals)];stored=np.r_[expected['per_fold_delta'],expected['delta_mean']]
   errors.extend(abs(observed-stored).tolist())
   bootstrap=result['bootstrap_results'][pair+'/'+condition]
   assert bootstrap['n_bootstrap']==20000 and bootstrap['seed']==20261108
   assert bootstrap['per_fold_delta']==expected['per_fold_delta'] and bootstrap['delta_mean']==expected['delta_mean']
   assert len(bootstrap['ci95_percentile'])==2 and np.isfinite(bootstrap['ci95_percentile']).all()
   mean_checks.append({'condition':condition,'pair':pair,'point_difference':float(observed[-1]),'stored_interval_not_recomputed':bootstrap['ci95_percentile']})
 assert len(errors)==24 and max(errors)<1e-12
 code=(ROOT/'v22_sources/secom_bootstrap_recompute_20261108.py').read_text();tree=ast.parse(code)
 methods=[(n.func.attr,n.lineno) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 writers=[(name,line) for name,line in methods if name in {'write_text','write_bytes','save','savez','savez_compressed','dump','tofile','savetxt'}]
 assert writers==[('write_text',237)],writers
 defaults={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):
   for k in n.keywords:
    if k.arg=='default':defaults[n.args[0].value]=ast.literal_eval(k.value)
 assert defaults['--n-boot']==20000 and defaults['--seed']==20261108
 with zipfile.ZipFile(ROOT/'original_zips/CCMS_Online_Resource_3_v2.1_路径修订版.zip') as z:
  same=[];different=[]
  for n in z.namelist():
   (same if z.read(n)==(PACKAGE/'frozen/original_or3'/n).read_bytes() else different).append(n)
  assert set(different)=={'README_Online_Resource_3.md','SHA256SUMS_Online_Resource_3.txt','r13_threeway_20k.json','reproduce_bootstrap_20k.py'}
  diffs=differences(json.loads((PACKAGE/'frozen/original_or3/r13_threeway_20k.json').read_text()),json.loads(z.read('r13_threeway_20k.json')))
  assert set(diffs)=={f'/sources/{i}/path' for i in range(6)}
  old=ast.parse((PACKAGE/'frozen/original_or3/reproduce_bootstrap_20k.py').read_text());new=ast.parse(z.read('reproduce_bootstrap_20k.py'))
  assert isinstance(old.body[0],ast.Expr) and isinstance(new.body[0],ast.Expr)
  old.body.pop(0);new.body.pop(0);assert ast.dump(old)==ast.dump(new)
  with np.load(io.BytesIO(z.read('aps_threeway_draws_20k.npz')),allow_pickle=False) as draws:
   assert len(draws.files)==24 and all(draws[n].shape==(20000,5) for n in draws.files)
 return {'status':'passed','archive_checks':archives,'archive_manifest_matches':total,'SECOM_source_hashes_matched':5,'SECOM_point_scalar_checks':24,'SECOM_point_max_error':max(errors),'SECOM_stored_results':mean_checks,'script_defaults':defaults,'persistent_data_write_calls':writers,'OR3_identical_members':same,'OR3_changed_members':different,'OR3_JSON_changes_only_six_source_paths':True,'OR3_script_executable_AST_identical_after_module_docstring_removed':True,'SECOM_draw_intervals_independently_recomputed':False,'SECOM_historical_execution_authenticated':False,'new_training':0,'new_bootstrap_draws':0,'remaining_gaps':['A3 runtime/prediction/model/mask/index originals','SECOM 20,000-draw arrays/weights/indices, independent execution logs and historical runtime evidence','unprovided current author facts']}
if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
