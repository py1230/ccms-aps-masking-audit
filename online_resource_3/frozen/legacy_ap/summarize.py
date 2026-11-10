from pathlib import Path
import sys,json,time
import numpy as np
from sklearn.metrics import average_precision_score as AP
R=Path(__file__).resolve().parents[1]/'results'
FAM={'S0':['S0'],'S1_30':['S1_30_101','S1_30_202','S1_30_303'],'S2':['S2_1101','S2_1202','S2_1303'],'S3':['S3_2101','S3_2202','S3_2303']}
def save(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def prep(y,p):
 o=np.argsort(-p,kind='stable');last=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(y)-1]
 return o,y[o],last

def wap(pr,w):
 o,y,last=pr;ww=w[o];tp=np.cumsum(ww*y)[last];den=np.cumsum(ww)[last];total=tp[-1]
 if total<=0:raise ValueError('zero bootstrap positives')
 return float(np.sum(np.diff(np.r_[0.,tp])*np.divide(tp,den,out=np.zeros_like(tp),where=den>0))/total)
def interval(v):return [float(z) for z in np.quantile(v,[.025,.975])]
def aps_boot():
 packs=[np.load(R/f'aps_seed{s}.npz') for s in [11,22,33,44,55]]
 out={'n_bootstrap':1000,'seed':20261101,'estimand':'mean per-split AP differences, first averaged within stress-family; fixed alpha0.25 ens3','results':{}}
 draws={}
 for dataset in ['val','test']:
  ys=[z['y_'+dataset] for z in packs]
  ids=[z['val_rows'] for z in packs] if dataset=='val' else [np.arange(len(ys[0])) for z in packs]
  union=np.unique(np.concatenate(ids));uy=np.empty(len(union),int)
  mappings=[np.searchsorted(union,v) for v in ids]
  for y,m in zip(ys,mappings):uy[m]=y
  pr={};points={};boots={f:np.zeros((1000,5)) for f in FAM}
  for f,conds in FAM.items():
   pr[f]=[[[(prep(y,z['C025__'+dataset+'__'+k]),prep(y,z['B025__'+dataset+'__'+k])) for k in conds] for z,y in zip(packs,ys)]][0]
   points[f]=[float(np.mean([AP(y,z['C025__'+dataset+'__'+k])-AP(y,z['B025__'+dataset+'__'+k]) for k in conds])) for z,y in zip(packs,ys)]
  rng=np.random.default_rng(20261101);pidx=np.flatnonzero(uy==1);nidx=np.flatnonzero(uy==0)
  for b in range(1000):
   ix=np.r_[rng.choice(pidx,len(pidx),replace=True),rng.choice(nidx,len(nidx),replace=True)]
   weights=np.bincount(ix,minlength=len(union)).astype(float)
   for f in FAM:
    for si,m in enumerate(mappings):
     w=weights[m];boots[f][b,si]=np.mean([wap(a,w)-wap(c,w) for a,c in pr[f][si]])
   if b%250==0:print('BOOT',dataset,b,flush=True)
  out['results'][dataset]={}
  for f in FAM:
   out['results'][dataset][f]={'delta_mean':float(np.mean(points[f])),'ci95':interval(boots[f].mean(1)),'per_split':[{'seed':s,'delta':p,'ci95':interval(boots[f][:,i])} for i,(s,p) in enumerate(zip([11,22,33,44,55],points[f]))]}
   draws[dataset+'__'+f]=boots[f]
  out[dataset+'_sampling']='unique original validation IDs stratified by label, shared weights for overlapping splits' if dataset=='val' else 'same official test row weights shared across all five fitted models and all stress inputs'
 np.savez_compressed(R/'aps_bootstrap_draws.npz',**draws);save('aps_matched_bootstrap.json',out)

def secom_summary():
 x=json.loads((R/'secom_corrected.json').read_text());fam={k:v for k,v in FAM.items() if k in ['S0','S1_30']};out={'per_config':{},'paired':{},'n_bootstrap':1000,'resample':'within outer-fold and label, mean of fold AP, fixed fitted models'}
 configs=['M3','M4','C025','B025','C_selected','B_selected']
 for c in configs:
  out['per_config'][c]={}
  for f,ks in fam.items():
   vals=[float(np.mean([z['metrics'][c][k] for k in ks])) for z in x['folds']]
   pool=float(np.mean([x['pooled'][c][k] for k in ks]))
   out['per_config'][c][f]={'fold_AP':vals,'mean_fold_AP':float(np.mean(vals)),'pooled_AP_secondary':pool}
 packs=[np.load(R/f'secom_fold{i}.npz') for i in range(5)]
 for label,ca,cb in [('fixed_C-B','C025','B025'),('selected_C-B','C_selected','B_selected')]:
  out['paired'][label]={}
  for f,ks in fam.items():
   d=np.empty((1000,5));points=[]
   for i,z in enumerate(packs):
    y=z['y_test'];pa=[z[ca+'__test__'+k] for k in ks];pb=[z[cb+'__test__'+k] for k in ks];prs=[(prep(y,a),prep(y,b)) for a,b in zip(pa,pb)]
    points.append(float(np.mean([AP(y,a)-AP(y,b) for a,b in zip(pa,pb)])))
    rng=np.random.default_rng(20261101+i);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    for bi in range(1000):
     ids=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];w=np.bincount(ids,minlength=len(y)).astype(float)
     d[bi,i]=np.mean([wap(a,w)-wap(b,w) for a,b in prs])
   out['paired'][label][f]={'delta_mean':float(np.mean(points)),'ci95':interval(d.mean(1)),'per_fold':points}
 save('secom_summary.json',out)

def mechanism():
 x=json.loads((R/'aps_reconstruction.json').read_text());X=None
 # Raw train/validation features determine buckets; trained predictions are reused.
 sys.path[:0]=[str(Path(__file__).resolve().parents[3] / '03_原始项目归档/CCMS核查_2026-10-31/src')]
 from aps_io import load_aps_csv,split_features
 X,y=split_features(load_aps_csv(str(Path(__file__).resolve().parents[3] / '04_实际使用的数据/aps/aps_failure_training_set.csv'))[0])
 rows=[]
 for s in x['splits']:
  z=np.load(R/f"aps_seed{s['seed']}.npz");tr,va=z['train_rows'],z['val_rows'];di=[q for q in s['deletion_diagnostics'] if q.get('arm')!='B'];B=[q for q in s['deletion_diagnostics'] if q.get('arm')=='B']
  # Top columns fixed from expected positive deletion, independent of validation labels.
  q=np.array(di[0]['gap_before']);visible=np.array(di[0]['visible_by_class_col']);n0=(y[tr]==0).sum();n1=(y[tr]==1).sum();rates=np.array([1-visible[0]/n0,1-visible[1]/n1]);expected_pos=.25*n1*(rates.max(0)-rates[1]);top=np.argsort(-expected_pos)[:8]
  nobs=(~np.isnan(X[va][:,top])).sum(1);buckets=[]
  for k in range(9):
   use=nobs==k
   if use.any():buckets.append({'observed':k,'n':int(use.sum()),'positive_rate':float(y[va][use].mean()),'mean_score_M3':float(z['M3__val__S0'][use].mean()),'mean_score_C025':float(z['C025__val__S0'][use].mean())})
  rows.append({'seed':s['seed'],'top8_columns':top.tolist(),'pos_deletion_top8_share':[float(np.array(q['deleted_by_class_col'])[1,top].sum()/np.array(q['deleted_by_class_col'])[1].sum()) for q in di],'C_deleted':[[int(np.array(q['deleted_by_class_col'])[c].sum()) for c in [0,1]] for q in di],'B_deleted':[[int(np.array(q['deleted_by_class_col'])[c].sum()) for c in [0,1]] for q in B],'buckets':buckets,'C_visible_deletion_rate':[float(np.array(q['deleted_by_class_col']).sum()/visible.sum()) for q in di],'B_visible_deletion_rate':[float(np.array(q['deleted_by_class_col']).sum()/visible.sum()) for q in B]})
 save('mechanism_alpha025.json',{'alpha':.25,'splits':rows})

def test_weighted():
 rng=np.random.default_rng(3)
 for n in [21,151]:
  y=np.r_[1,np.zeros(n-2),1];p=rng.integers(0,4,n)/4;w=rng.integers(0,5,n).astype(float);w[0]=1
  assert abs(wap(prep(y,p),w)-AP(y,p,sample_weight=w))<1e-12
 print('weighted AP tie/zero-weight checks PASS',flush=True)
if __name__=='__main__':
 test_weighted();{'aps':aps_boot,'secom':secom_summary,'mechanism':mechanism}[sys.argv[1]]()
