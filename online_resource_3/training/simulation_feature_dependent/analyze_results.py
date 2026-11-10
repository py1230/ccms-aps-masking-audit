"""Frozen seed-level analysis. No training, selection or target-dependent choices.

Runtime: same SciPy/NumPy environment as training; no pandas/plot dependency.
Input: summary.json produced by run_simulation.py. Outputs fail closed on missing seeds.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SEEDS = list(range(701,731))
MULTIPLIERS = [1,.75,.5,.25,0,-.25,-.5,-.75,-1]
CONDITIONS = [f'gap_{m:g}' for m in MULTIPLIERS]
CONFIGS = ['M3','M4','C025','B025','C050','B050','C100','B100','C_selected','B_selected']
BOOTSTRAP_SEED = 20261109
BOOTSTRAP_DRAWS = 20000
PRECISION_TARGET = .005

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(x, infer=False):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Finite complete seed vectors required')
    n=len(x);mean=float(x.mean());sd=0. if np.ptp(x)==0 else float(x.std(ddof=1));se=sd/math.sqrt(n)
    half=float(stats.t.ppf(.975,n-1)*se)
    d={'n':n,'mean':mean,'sd':sd,'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),
       'se':se,'ci95_t_low':mean-half,'ci95_t_high':mean+half,'ci95_t_halfwidth':half,
       'negative_n':int((x<0).sum()),'zero_n':int((x==0).sum()),'positive_n':int((x>0).sum())}
    if infer:
        if sd==0:
            test_t=0. if mean==0 else None;p=1. if mean==0 else 0.;dz=0. if mean==0 else None
        else:
            result=stats.ttest_1samp(x,0,alternative='two-sided',nan_policy='raise')
            test_t=float(result.statistic);p=float(result.pvalue);dz=mean/sd
        d.update(t=test_t,df=n-1,p_nominal=p,paired_dz=dz,zero_variance=sd==0)
    return d

def holm(ps):
    p=np.asarray(ps,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid p values')
    order=np.argsort(p,kind='stable'); out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def diagnostic(x,name):
    x=np.asarray(x);sd=0. if np.ptp(x)==0 else float(x.std(ddof=1));n=len(x)
    if sd==0:
        sw={'shapiro_w':None,'shapiro_p':None,'constant_vector':True}
    else:
        w,p=stats.shapiro(x);sw={'shapiro_w':float(w),'shapiro_p':float(p),'constant_vector':False}
    qq=stats.probplot(x,dist='norm',fit=False)
    counts,edges=np.histogram(x,bins=8)
    qrows=[{'comparison':name,'rank':i+1,'normal_quantile':float(q),'observed':float(v)} for i,(q,v) in enumerate(zip(*qq))]
    hrows=[{'comparison':name,'bin':i+1,'left':float(edges[i]),'right':float(edges[i+1]),'count':int(v)} for i,v in enumerate(counts)]
    return {'comparison':name,**sw,'method_switch_from_diagnostic':False},qrows,hrows

def analyze(input_path,output_dir):
    d=json.loads(input_path.read_text())
    if d.get('status')!='complete':raise ValueError('Refuse final analysis of incomplete run')
    seeds=d['seeds'];actual=[int(s['data_seed']) for s in seeds]
    if sorted(actual)!=SEEDS or len(set(actual))!=len(SEEDS):raise ValueError(f'Expected exactly seeds {SEEDS}; found {actual}')
    seeds=sorted(seeds,key=lambda s:s['data_seed'])
    ap=np.array([[[s['metrics'][c][condition] for condition in CONDITIONS] for c in CONFIGS] for s in seeds],float)
    if ap.shape!=(30,10,9) or not np.isfinite(ap).all() or np.any((ap<0)|(ap>1)):raise ValueError('Invalid AP tensor')
    fixed=ap[:,CONFIGS.index('C025')]-ap[:,CONFIGS.index('B025')]
    selected=ap[:,CONFIGS.index('C_selected')]-ap[:,CONFIGS.index('B_selected')]
    primary=fixed.mean(axis=1)
    # Column order primary, fixed9, selected9; shared complete seed-index resampling.
    vectors=np.column_stack([primary,fixed,selected])
    ix=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(SEEDS),size=(BOOTSTRAP_DRAWS,len(SEEDS)))
    boot=vectors[ix].mean(axis=1)
    quantiles=np.quantile(boot,[.025,.975],axis=0,method='linear')
    primary_summary={'estimand':'equal-weight mean over nine fixed m conditions of seed-paired C025 minus B025 AP',**describe(primary),
        'bootstrap_percentile_low':float(quantiles[0,0]),'bootstrap_percentile_high':float(quantiles[1,0]),
        'precision_halfwidth_target':PRECISION_TARGET,'precision_met':bool(describe(primary)['ci95_t_halfwidth']<=PRECISION_TARGET),
        'leave_one_seed_out_mean_min':float(((primary.sum()-primary)/29).min()),
        'leave_one_seed_out_mean_max':float(((primary.sum()-primary)/29).max()),
        'formal_primary_zero_test':False}
    comparisons=[]
    for fam,arr,offset in [('fixed_C025_minus_B025',fixed,1),('selected_C_minus_B',selected,10)]:
        for j,m in enumerate(MULTIPLIERS):
            comparisons.append({'comparison':fam,'m':m,**describe(arr[:,j],infer=True),
                'bootstrap_percentile_low':float(quantiles[0,offset+j]),'bootstrap_percentile_high':float(quantiles[1,offset+j])})
    corrected=holm([r['p_nominal'] for r in comparisons])
    for row,p in zip(comparisons,corrected):row.update(p_holm18=float(p),reject_holm18_05=bool(p<.05))
    ap_rows=[{'config':c,'m':m,**describe(ap[:,i,j])} for i,c in enumerate(CONFIGS) for j,m in enumerate(MULTIPLIERS)]
    by_seed=[{'seed':s,'primary_nine_point_mean':float(primary[k]),**{f'{fam}__gap_{m:g}':float(arr[k,j]) for fam,arr in [('fixed',fixed),('selected',selected)] for j,m in enumerate(MULTIPLIERS)}} for k,s in enumerate(SEEDS)]
    ap_seed=[{'seed':s,'config':c,'m':m,'ap':float(ap[k,i,j])} for k,s in enumerate(SEEDS) for i,c in enumerate(CONFIGS) for j,m in enumerate(MULTIPLIERS)]
    selections=[{'seed':s['data_seed'],'arm':arm,'alpha':s['source_selection'][arm]['chosen']['alpha'],'config':s['source_selection'][arm]['chosen']['config']} for s in seeds for arm in ['C','B']]
    counts={arm:{str(a):sum(r['arm']==arm and float(r['alpha'])==a for r in selections) for a in [0,.25,.5,1]} for arm in ['C','B']}
    diags=[];qq=[];hist=[]
    names=['primary']+[f'{fam}__gap_{m:g}' for fam in ['fixed','selected'] for m in MULTIPLIERS]
    for j,name in enumerate(names):
        diag,q,h=diagnostic(vectors[:,j],name);diags.append(diag);qq.extend(q);hist.extend(h)
    output_dir.mkdir(parents=True,exist_ok=False)
    write_json(output_dir/'statistical_summary.json',{'status':'complete','input_sha256':sha(input_path),'analysis_script_sha256':sha(__file__),
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'planned_seeds':SEEDS,'completed_seeds':SEEDS,'n_independent_units':30,
        'bootstrap':{'draws':BOOTSTRAP_DRAWS,'rng_seed':BOOTSTRAP_SEED,'unit':'entire independent seed vector','interval':'pointwise percentile; sensitivity only'},
        'primary':primary_summary,'secondary_comparisons':comparisons,'selection_counts':counts,
        'multiplicity':'18 secondary paired t comparisons in one Holm family; primary CI not simultaneous with this family',
        'interpretation':'fixed simulation generator and equal grid weights; not deployment prevalence or a deployment threshold'})
    for name,rows in [('secondary_comparisons',comparisons),('ap_summary',ap_rows),('contrasts_by_seed',by_seed),('ap_by_seed',ap_seed),('selections',selections),('normality_diagnostics',diags),('qq_plot_data',qq),('histogram_data',hist)]:write_csv(output_dir/(name+'.csv'),rows)
    np.savez_compressed(output_dir/'bootstrap_seed_indices.npz',indices=ix,data_seed_order=np.array(SEEDS),vector_column_names=np.array(names),bootstrap_means=boot)
    write_json(output_dir/'analysis_inputs.json',{'input':str(input_path),'sha256':sha(input_path),'seed_order':SEEDS,'config_order':CONFIGS,'condition_order':CONDITIONS})
    print(json.dumps({'primary':primary_summary,'selection_counts':counts,'holm18_rejections':[{'comparison':r['comparison'],'m':r['m'],'p':r['p_holm18']} for r in comparisons if r['reject_holm18_05']]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE/'summary.json');p.add_argument('--output-dir',type=Path,default=HERE/'analysis');a=p.parse_args();analyze(a.input.resolve(),a.output_dir.resolve())
