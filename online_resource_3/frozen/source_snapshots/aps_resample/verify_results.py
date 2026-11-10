"""Independent verifier for empirical training-ID bootstrap; never imports runner.

Subcommands: --preflight; --unit anchor/seed90101; --final.
Model reloading checks artifacts and arithmetic, not a second training run.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import argparse,csv,hashlib,json,platform,time,warnings
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib,sklearn
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
VAL=['S0','S1_10_101','S1_10_202','S1_10_303']
TEST=VAL+[f'S1_30_{s}' for s in [101,202,303]]+[f'S2_{s}' for s in [1101,1202,1303]]+[f'S3_{s}' for s in [2101,2202,2303]]
GROUPS=['S0','S1_10','S1_30','S2','S3']
CONTRASTS=[('C_selected','B_selected'),('C_selected','BASE'),('B_selected','BASE'),('C025','B025'),('C025','BASE'),('B025','BASE')]
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def raw(p):
 with Path(p).open() as f:
  for skip,line in enumerate(f):
   if line.startswith('class,'):break
  else:raise ValueError('CSV header absent')
 df=pd.read_csv(p,skiprows=skip,na_values=['na'],low_memory=False);labels=df.pop('class');assert labels.isin(['pos','neg']).all()
 x=df.to_numpy(dtype=np.float64);assert np.isfinite(x[~np.isnan(x)]).all()
 return x,(labels.to_numpy()=='pos').astype(np.int64),np.asarray(df.columns)
def load():
 x,y,f=raw(HERE/'raw_data/aps_failure_training_set.csv');xt,yt,ft=raw(HERE/'raw_data/aps_failure_test_set.csv')
 assert x.shape==(60000,170) and xt.shape==(16000,170) and int(y.sum())==1000 and int(yt.sum())==375 and np.array_equal(f,ft)
 cfg=read(HERE/'config.json');assert cfg['input_environment']=={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}
 assert cfg['bootstrap_seeds']==list(range(90101,90121)) and cfg['alphas']==[0,.25,.5,1.] and cfg['mask_seeds']==[19107,19108,19109]
 assert cfg['model_params']['early_stopping'] is False and cfg['model_params']['max_iter']==100 and cfg['model_params']['random_state']==2026
 assert cfg['selection_conditions']==VAL and cfg['test_conditions']==TEST and cfg['refit_after_selection'] is False
 for name,rec in read(HERE/'input_sources.json').items():assert sha(HERE/name)==rec['sha256']
 return cfg,x,y,xt,yt,f,npz(HERE/'split_indices.npz')
def close(a,b,tol=1e-12):
 a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(d,tol);return d
def ap(y,p):
 y=np.asarray(y);p=np.asarray(p);order=np.argsort(-p,kind='stable');s=p[order];t=y[order]
 ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(t,dtype=np.int64)[ends]
 return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/(ends+1)))
def median(x):
 with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);m=np.nanmedian(x,axis=0)
 return np.where(np.isfinite(m),m,0.)
def matrix(x,m):
 missing=np.isnan(x);return np.column_stack([np.where(missing,m,x),missing.astype(float)])
def stress(x,c):
 v=~np.isnan(x);out=np.zeros(x.shape,dtype=bool)
 if c=='S0':return out
 p=c.split('_');rng=np.random.default_rng(int(p[-1]))
 if p[0]=='S1':return v&(rng.random(x.shape)<int(p[1])/100)
 if p[0]=='S2':
  rows=rng.choice(len(x),size=int(round(.5*len(x))),replace=False);out[rows]=v[rows]&(rng.random((len(rows),x.shape[1]))<.6)
 elif p[0]=='S3':
  cols=rng.choice(x.shape[1],size=max(1,int(round(.3*x.shape[1]))),replace=False);out[:,cols]=v[:,cols]
 else:raise ValueError(c)
 return out
def quantities(x,y):
 counts=np.bincount(y,minlength=2);missing=np.asarray([np.sum(np.isnan(x[y==c]),axis=0) for c in [0,1]],dtype=np.int64);q=missing/counts[:,None]
 return counts,missing,q
def trainmask(x,y,arm,alpha,seed,q,counts):
 if arm=='BASE':return np.zeros(x.shape,dtype=bool)
 mx=np.max(q,axis=0)
 if arm=='C':
  p=np.zeros_like(q);np.divide(alpha*(mx-q),1-q,out=p,where=q<1);p=np.clip(p,0,1)[y]
 else:
  qbar=(counts/counts.sum())@q;p=np.zeros(x.shape[1]);np.divide(alpha*(mx-qbar),1-qbar,out=p,where=qbar<1);p=np.clip(p,0,1)
 return (~np.isnan(x))&(np.random.default_rng(seed).random(x.shape)<p)
def specs():
 ans=[('BASE','BASE',0.,None)]
 for a in [.25,.5,1.]:
  for arm in ['C','B']:
   for seed in [19107,19108,19109]:ans.append((f'{arm}{round(100*a):03d}_seed{seed}',arm,a,seed))
 return ans
def grouped(values):return {g:float(values['S0']) if g=='S0' else float(np.mean([v for k,v in values.items() if k.startswith(g+'_')])) for g in GROUPS}
def ids(unit,cfg,y,ix):
 stored=npz(HERE/'prepared_indices'/f'{unit}.npz');meta=read(HERE/'prepared_indices'/f'{unit}.json');tr=ix['s11_train'];va=ix['s11_val'];rs=ix['s11_reserve']
 assert np.array_equal(np.sort(np.r_[tr,va,rs]),np.arange(60000)) and [len(tr),len(va),len(rs)]==[42000,9000,9000]
 assert np.array_equal(stored['base_train_rows'],tr) and np.array_equal(stored['val_rows'],va) and np.array_equal(stored['reserve_rows'],rs)
 seed=None if unit=='anchor' else int(unit[4:]);assert meta['seed']==seed and meta['unit']==unit
 if seed is None:draw=tr.copy();assert meta['rng_states']=={}
 else:
  assert seed in cfg['bootstrap_seeds'];rng=np.random.default_rng(seed);assert rng.bit_generator.state==meta['rng_states']['before']
  neg=rng.choice(tr[y[tr]==0],41300,replace=True);pos=rng.choice(tr[y[tr]==1],700,replace=True)
  assert rng.bit_generator.state==meta['rng_states']['before_permutation'];draw=rng.permutation(np.r_[neg,pos]);assert rng.bit_generator.state==meta['rng_states']['after']
 assert np.array_equal(draw,stored['train_rows']);mult=np.bincount(draw,minlength=60000);assert np.array_equal(mult,stored['multiplicity'])
 assert len(draw)==42000 and np.isin(draw,tr).all() and not np.intersect1d(draw,va).size and not np.intersect1d(draw,rs).size
 assert np.bincount(y[draw],minlength=2).tolist()==meta['class_counts']==[41300,700]
 assert meta['rows_sha256']==ash(draw) and meta['multiplicity_sha256']==ash(mult)
 assert meta['unique_rows']==np.count_nonzero(mult) and meta['duplicate_occurrences']==len(draw)-np.count_nonzero(mult)
 return stored,{'unit':unit,'seed':seed,'unique_rows':int(np.count_nonzero(mult)),'unique_by_class':[int(np.count_nonzero(mult[y==c])) for c in [0,1]],'maximum_multiplicity':int(mult.max()),'out_of_bag_training_rows':int(np.count_nonzero(mult[tr]==0)),'duplicate_occurrences':int(len(draw)-np.count_nonzero(mult)),'rows_sha256':ash(draw),'multiplicity_sha256':ash(mult)}
def preflight():
 cfg,x,y,xt,yt,f,ix=load();names=['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']];rows=[]
 for unit in names:rows.append(ids(unit,cfg,y,ix)[1])
 for ds,z,conditions in [('validation',x[ix['s11_val']],VAL),('test',xt,TEST)]:
  saved=npz(HERE/f'{ds}_masks.npz');assert set(saved)==set(conditions)
  for c in conditions:assert np.array_equal(saved[c],stress(z,c))
 for yy,pp,target in [([1,0],[.5,.5],.5),([1,0,1,0],[.9,.8,.7,.6],5/6),([0,1,1,0],[.9,.8,.8,.7],2/3)]:assert abs(ap(yy,pp)-target)<1e-14
 result={'status':'passed','timestamp':stamp(),'verifier_sha256':sha(__file__),'config_sha256':sha(HERE/'config.json'),'prepared_units':21,'fixed_evaluation_masks':17,'units':rows,'scope':'record-ID isolation only; unknown truck grouping is not verified'}
 save(HERE/'verification_preflight.json',result);print(json.dumps({'status':'passed','stage':'preflight','units':21}),flush=True)

def verify_unit(unit):
 start=time.perf_counter();cfg,x,y,xt,yt,features,ix=load();root=HERE/'units'/unit;summary=read(root/'summary.json')
 assert summary['status']=='complete' and summary['fits']==19 and summary['unit']==unit and not (root/'failure.json').exists()
 frozen=read(HERE/'freeze.json')
 for name,h in frozen['files'].items():assert sha(HERE/name)==h,name
 index,idrow=ids(unit,cfg,y,ix);tr=index['train_rows'];va=index['val_rows'];xx=x[tr];yy=y[tr];med=median(xx);counts,missing,q=quantities(xx,yy)
 pre=npz(root/'preprocessing.npz')
 for k,v in [('medians',med),('q',q),('class_counts',counts),('missing_counts',missing),('train_rows',tr),('val_rows',va),('feature_names',features)]:assert np.array_equal(pre[k],v),k
 inp=read(root/'input_hashes.json')
 for k,v in [('X_train_raw',xx),('y_train',yy),('train_rows',tr),('X_val_raw',x[va]),('y_val',y[va]),('X_test_raw',xt),('y_test',yt),('median',med),('q',q)]:assert inp[k]==ash(v),k
 assert inp['config_sha256']==sha(HERE/'config.json') and inp['freeze_sha256']==sha(HERE/'freeze.json')
 em={};ehash=read(root/'evaluation_matrix_hashes.json')
 for ds,z,cond,filename in [('val',x[va],VAL,'validation_masks.npz'),('test',xt,TEST,'test_masks.npz')]:
  em[ds]=npz(HERE/filename);assert set(em[ds])==set(cond)
  for c in cond:
   assert np.array_equal(em[ds][c],stress(z,c));v=z.copy();v[em[ds][c]]=np.nan;assert ehash[ds+'__'+c]==ash(matrix(v,med))
 allpred={'val':{},'test':{}};maxprob=0.;maxap=0.;maxbudget=0.;arrays=0;records=read(root/'fit_records.json');assert len(records)==19
 for (tag,arm,a,seed),rec in zip(specs(),records):
  assert rec['tag']==tag;mask=trainmask(xx,yy,arm,a,seed,q,counts);saved=npz(root/'training_masks'/f'{tag}.npz')['new_mask'];assert np.array_equal(mask,saved) and not (mask&np.isnan(xx)).any()
  diag=npz(root/'diagnostics'/f'{tag}.npz')
  for k,v in [('row_deleted',mask.sum(1)),('col_deleted',mask.sum(0)),('class_col_deleted',np.stack([mask[yy==c].sum(0) for c in [0,1]]))]:assert np.array_equal(diag[k],v),k
  qbar=(counts/counts.sum())@q;p=np.zeros(170);np.divide(a*(q.max(0)-qbar),1-qbar,out=p,where=qbar<1)
  expected_c=a*np.sum(counts[:,None]*(q.max(0)-q),axis=0);expected_b=p*np.sum(~np.isnan(xx),axis=0)
  maxbudget=max(maxbudget,close(expected_c,expected_b,1e-8))
  xm=xx.copy();xm[mask]=np.nan;train=matrix(xm,med);tsha=ash(train);del train,xm
  obj=joblib.load(root/'models'/f'{tag}.joblib');m=obj['model']
  assert m.get_params()==cfg['model_params']==rec['effective_params'] and m.n_features_in_==340 and m.n_iter_==100 and not m.do_early_stopping_
  assert rec['n_iter']==100 and rec['do_early_stopping'] is False
  for k,v in [('unit',unit),('arm',arm),('alpha',a),('mask_seed',seed),('model_seed',2026),('channel','M3')]:assert obj[k]==v,k
  assert np.array_equal(obj['medians'],med) and np.array_equal(obj['train_rows'],tr)
  assert obj['training_matrix_sha256']==rec['training_matrix_sha256']==tsha and obj['training_labels_sha256']==ash(yy)
  assert obj['mask_sha256']==rec['mask_sha256']==ash(mask) and obj['config_sha256']==sha(HERE/'config.json') and rec['model_sha256']==sha(root/'models'/f'{tag}.joblib')
  for ds,z,labels,rows,conditions in [('val',x[va],y[va],va,VAL),('test',xt,yt,np.arange(len(yt)),TEST)]:
   saved=npz(root/'predictions'/f'{tag}__{ds}.npz');assert set(saved)==set(conditions+['row_indices','y'])
   assert np.array_equal(saved['row_indices'],rows) and np.array_equal(saved['y'],labels)
   for c in conditions:
    v=z.copy();v[em[ds][c]]=np.nan;pred=m.predict_proba(matrix(v,med))[:,1]
    assert ((saved[c]>=0)&(saved[c]<=1)).all();maxprob=max(maxprob,close(saved[c],pred));arrays+=1
   allpred[ds][tag]={c:saved[c] for c in conditions}
  del m,obj,mask
  print(json.dumps({'event':'model_verified','unit':unit,'tag':tag,'probability_arrays':arrays}),flush=True)
 ensemble=npz(root/'ensemble_predictions.npz');metrics=read(root/'metrics.json');own={}
 for k,v in [('y_val',y[va]),('y_test',yt),('train_rows',tr),('val_rows',va),('test_rows',np.arange(len(yt)))]:assert np.array_equal(ensemble[k],v)
 configs=['BASE','C025','B025','C050','B050','C100','B100'];assert set(metrics)==set(configs)
 for name in configs:
  own[name]={};tags=[t for t,_,_,_ in specs() if t.split('_seed')[0]==name]
  for ds,labels,conditions in [('val',y[va],VAL),('test',yt,TEST)]:
   for c in conditions:
    pred=np.mean([allpred[ds][t][c] for t in tags],axis=0);maxprob=max(maxprob,close(ensemble[name+'__'+ds+'__'+c],pred));score=ap(labels,pred);own[name][ds+'__'+c]=score;maxap=max(maxap,close(score,metrics[name][ds+'__'+c]))
 selection=read(root/'selection.json');choices={}
 for arm in ['C','B']:
  cand=[]
  for a in cfg['alphas']:
   name='BASE' if a==0 else f'{arm}{round(100*a):03d}';row={'alpha':a,'config':name,'s0':own[name]['val__S0'],'worst4':min(own[name]['val__'+c] for c in VAL)}
   row['eligible']=row['s0']>=own['BASE']['val__S0']-.01;cand.append(row)
  # Use archived full-precision AP to apply exact eligibility/ties; independent AP is tolerance checked above.
  expected=[]
  for a in cfg['alphas']:
   name='BASE' if a==0 else f'{arm}{round(100*a):03d}';s0=metrics[name]['val__S0'];expected.append({'alpha':a,'config':name,'s0':s0,'worst4':min(metrics[name]['val__'+c] for c in VAL),'eligible':s0>=metrics['BASE']['val__S0']-.01})
  assert selection['candidates'][arm]==expected
  eligible=[r for r in expected if r['eligible']];best=max(r['worst4'] for r in eligible);winner=min([r for r in eligible if r['worst4']==best],key=lambda r:r['alpha'])
  assert selection['chosen'][arm]==winner;choices[arm]=winner
  independently_eligible=[r for r in cand if r['eligible']];best2=max(r['worst4'] for r in independently_eligible)
  assert abs(winner['worst4']-best2)<=1e-12
  ownwinner=min([r for r in independently_eligible if r['worst4']==best2],key=lambda r:r['alpha'])
  assert ownwinner['config']==winner['config'],('independent exact-score winner differs',unit,arm,ownwinner,winner)
 assert selection['selected_before_test_prediction'] is True
 events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()];se=[i for i,r in enumerate(events) if r['event']=='selection_frozen'];assert len(se)==1
 assert sum(r['event']=='fit' for r in events[:se[0]])==19 and events[se[0]]['selection_sha256']==sha(root/'selection.json')
 starts=[i for i,r in enumerate(events) if r['event']=='test_prediction_started'];saved_events=[i for i,r in enumerate(events) if r['event']=='test_prediction_saved']
 assert len(starts)==1 and starts[0]>se[0] and len(saved_events)==19 and min(saved_events)>starts[0]
 assert events[starts[0]]['selection_sha256']==sha(root/'selection.json')
 assert [events[i]['tag'] for i in saved_events]==[r[0] for r in specs()]
 for i in saved_events:
  event=events[i];assert event['selection_sha256']==sha(root/'selection.json') and event['prediction_sha256']==sha(root/'predictions'/f"{event['tag']}__test.npz")
 assert events[-1]['event']=='complete' and events[-1]['fits']==19
 gp={name:grouped({c:own[name]['test__'+c] for c in TEST}) for name in configs}
 for arm in ['C','B']:gp[arm+'_selected']=gp[choices[arm]['config']]
 contrasts={a+'-'+b:{g:gp[a][g]-gp[b][g] for g in GROUPS} for a,b in CONTRASTS}
 for name,vals in gp.items():
  for g,v in vals.items():maxap=max(maxap,close(v,summary['ap'][name][g]))
 for name,vals in contrasts.items():
  for g,v in vals.items():maxap=max(maxap,close(v,summary['contrasts'][name][g]))
 assert summary['chosen_alphas']=={a:r['alpha'] for a,r in choices.items()}
 result={'status':'passed','timestamp':stamp(),'unit':unit,'verifier_sha256':sha(__file__),'freeze_sha256':sha(HERE/'freeze.json'),'config_sha256':sha(HERE/'config.json'),'models':19,'training_masks':19,'probability_arrays':arrays,'candidate_ensemble_arrays':119,'candidate_ap_checks':119,'selection_arms':2,'max_prediction_error':maxprob,'max_ap_error':maxap,'max_expected_budget_error':maxbudget,'ids':idrow,'own_summary':{'unit':unit,'bootstrap_seed':None if unit=='anchor' else int(unit[4:]),'chosen_alphas':{a:r['alpha'] for a,r in choices.items()},'ap':gp,'contrasts':contrasts},'seconds':time.perf_counter()-start,'unit_file_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
 assert arrays==323
 out=HERE/'verification_units';out.mkdir(exist_ok=True);assert not (out/f'{unit}.json').exists();save(out/f'{unit}.json',result)
 print(json.dumps({'status':'passed','unit':unit,'seconds':result['seconds'],'max_prediction_error':maxprob,'max_ap_error':maxap}),flush=True)

def final():
 cfg=read(HERE/'config.json');frozen=read(HERE/'freeze.json')
 for name,h in frozen['files'].items():assert sha(HERE/name)==h,name
 names=['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']];verified=[]
 for name in names:
  rec=read(HERE/'verification_units'/f'{name}.json');assert rec['status']=='passed' and rec['unit']==name and rec['models']==19
  assert rec['verifier_sha256']==sha(__file__) and rec['freeze_sha256']==sha(HERE/'freeze.json') and rec['config_sha256']==sha(HERE/'config.json')
  for path,digest in rec['unit_file_sha256'].items():assert sha(HERE/path)==digest,path
  verified.append(rec)
 rows=[r['own_summary'] for r in verified];anchor=rows[0];reps=rows[1:];assert len(reps)==20
 stats=[];error=0.
 for kind in ['ap','contrasts']:
  for measure in anchor[kind]:
   for group in GROUPS:
    values=[r[kind][measure][group] for r in reps];mean=float(np.sum(values)/20);sd=float(np.sqrt(np.sum((np.asarray(values)-mean)**2)/19));base=anchor[kind][measure][group]
    stats.append({'kind':kind,'measure':measure,'group':group,'n':20,'mean':mean,'sample_sd':sd,'min':min(values),'max':max(values),'anchor':base,'mean_minus_anchor':mean-base,'values':values})
 saved=read(HERE/'descriptive_summary.json');assert saved['status']=='complete' and saved['formal_fits']==399 and saved['anchor_fits']==19 and saved['bootstrap_fits']==380 and saved['bootstrap_repeats']==20
 assert len(saved['results'])==len(stats)==75
 for row in stats:
  match=[r for r in saved['results'] if all(r[k]==row[k] for k in ['kind','measure','group'])];assert len(match)==1
  for k in ['n','mean','sample_sd','min','max','anchor','mean_minus_anchor','values']:error=max(error,close(row[k],match[0][k]))
 counts={a:{str(alpha):sum(r['chosen_alphas'][a]==alpha for r in reps) for alpha in cfg['alphas']} for a in ['C','B']}
 assert counts==saved['selection_counts'] and anchor['chosen_alphas']==saved['anchor_choices']
 for tag,measure in [('primary','C_selected-B_selected'),('key_secondary','C025-B025')]:
  found=next(r for r in stats if r['kind']=='contrasts' and r['measure']==measure and r['group']=='S1_30')
  for k in ['n','mean','sample_sd','min','max','anchor','mean_minus_anchor','values']:error=max(error,close(found[k],saved[tag][k]))
 expected={};expected['ap_by_unit.csv']=[{'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'config':c,'group':g,'AP':v} for r in rows for c,values in r['ap'].items() for g,v in values.items()]
 expected['contrasts_by_unit.csv']=[{'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'contrast':c,'group':g,'difference':v} for r in rows for c,values in r['contrasts'].items() for g,v in values.items()]
 expected['selections.csv']=[{'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'C_alpha':r['chosen_alphas']['C'],'B_alpha':r['chosen_alphas']['B']} for r in rows]
 expected['descriptive_summary.csv']=stats
 for filename,keys,values in [('ap_by_unit.csv',['unit','config','group'],['AP']),('contrasts_by_unit.csv',['unit','contrast','group'],['difference']),('selections.csv',['unit'],['C_alpha','B_alpha']),('descriptive_summary.csv',['kind','measure','group'],['n','mean','sample_sd','min','max','anchor','mean_minus_anchor'])]:
  with (HERE/filename).open() as f:got=list(csv.DictReader(f))
  want=expected[filename];assert len(got)==len(want)
  for row in want:
   match=[r for r in got if all(str(row[k])==r[k] for k in keys)];assert len(match)==1
   for k in values:error=max(error,close(row[k],float(match[0][k])))
   if 'bootstrap_seed' in row:assert match[0]['bootstrap_seed']==('' if row['bootstrap_seed'] is None else str(row['bootstrap_seed']))
 launcher=read(HERE/'launcher_receipt.json');assert launcher['status']=='complete' and len(launcher['records'])==21
 assert all(r['exit_code']==0 and r['status']=='complete' for r in launcher['records'])
 result={'status':'passed','timestamp':stamp(),'scope':'independent raw-data, resampling, mask, saved-model prediction and descriptive-statistic reconstruction; not independent retraining, population inference or historical provenance','verifier_sha256':sha(__file__),'freeze_sha256':sha(HERE/'freeze.json'),'config_sha256':sha(HERE/'config.json'),'formal_models':sum(r['models'] for r in verified),'anchor_models':19,'bootstrap_models':380,'bootstrap_repeats':20,'training_masks':sum(r['training_masks'] for r in verified),'probability_arrays':sum(r['probability_arrays'] for r in verified),'candidate_ensemble_arrays':sum(r['candidate_ensemble_arrays'] for r in verified),'candidate_ap_checks':sum(r['candidate_ap_checks'] for r in verified),'selection_arms':sum(r['selection_arms'] for r in verified),'max_prediction_error':max(r['max_prediction_error'] for r in verified),'max_ap_error':max(r['max_ap_error'] for r in verified),'max_expected_budget_error':max(r['max_expected_budget_error'] for r in verified),'max_summary_error':error,'summary_statistics':75,'summary_csv_rows':{k:len(v) for k,v in expected.items()},'selection_counts':counts,'anchor_choices':anchor['chosen_alphas'],'primary':next(r for r in stats if r['kind']=='contrasts' and r['measure']=='C_selected-B_selected' and r['group']=='S1_30'),'key_secondary':next(r for r in stats if r['kind']=='contrasts' and r['measure']=='C025-B025' and r['group']=='S1_30'),'unit_evidence_sha256':{name:sha(HERE/'verification_units'/f'{name}.json') for name in names},'summary_sha256':{name:sha(HERE/name) for name in ['descriptive_summary.json',*expected]},'unit_identity_diagnostics':[r['ids'] for r in verified],'all_models_100_iterations_and_no_early_stopping':True,'all_unit_selection_persisted_before_test_prediction':True}
 assert result['formal_models']==399 and result['probability_arrays']==6783 and result['candidate_ap_checks']==2499
 save(HERE/'verification.json',result);print(json.dumps({'status':'passed','formal_models':399,'primary':result['primary'],'key_secondary':result['key_secondary'],'selection_counts':counts}),flush=True)

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--unit');g.add_argument('--final',action='store_true');a=p.parse_args()
 with threadpool_limits(limits=2):
  if a.preflight:preflight()
  elif a.unit:verify_unit(a.unit)
  else:final()
if __name__=='__main__':main()
