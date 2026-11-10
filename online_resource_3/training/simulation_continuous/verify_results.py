"""Independent artifact verifier: never imports experiment or analysis implementation.
Uses NumPy generation arithmetic, an independent grouped-threshold AP, and trusted
newly fitted sklearn model reloads; no fitting is performed here.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse, hashlib, json, math, traceback
from datetime import datetime,timezone
import joblib,numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
MS=[1,.75,.5,.25,0,-.25,-.5,-.75,-1]
GAPS=[f'gap_{m:g}' for m in MS]
VALS=['S0','S1_10_101','S1_10_202','S1_10_303']
TAGS=['M3','M4']+[f'{arm}{a:03d}_seed{s}' for a in [25,50,100] for arm in ['C','B'] for s in [9107,9108,9109]]
BASE_CONFIGS=['M3','M4']+[f'{arm}{a:03d}' for a in [25,50,100] for arm in ['C','B']]
CONFIGS=BASE_CONFIGS+['C_selected','B_selected']
COUNTS={'seeds':0,'models_reloaded':0,'single_prediction_arrays':0,'ensemble_prediction_arrays':0,'ap_values':0,'selections':0,'training_masks':0,'file_hashes':0}
ERRORS={'prediction_max_abs':0.,'ensemble_max_abs':0.,'ap_max_abs':0.}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def eq(a,b,label):
    assert np.array_equal(a,b,equal_nan=True),label

def close(a,b,label,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,label
    assert np.isfinite(a).all() and np.isfinite(b).all(),label
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    assert err<=tol,(label,err)
    return err

def ap(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float)
    assert y.shape==p.shape and set(np.unique(y))=={0,1} and np.isfinite(p).all()
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    endpoints=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
    tp=np.cumsum(ys)[endpoints];precision=tp/(endpoints+1)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())

def reload_input(x,median,tag):
    if tag=='M4':return x
    z=x.copy();rr,cc=np.where(np.isnan(z));z[rr,cc]=median[cc]
    return np.concatenate((z,np.isnan(x).astype(float)),axis=1)

def independent_generation(seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(8000,20));score=1.2*(X[:,0]+X[:,1]+X[:,2]);lo,hi=-20.,20.
    for _ in range(80):
        mid=(lo+hi)/2
        if (1/(1+np.exp(-(score-mid)))).mean()>.05:lo=mid
        else:hi=mid
    u=rng.random(8000);intercept=(lo+hi)/2;y=(u<1/(1+np.exp(-(score-intercept)))).astype(int)
    r=np.random.default_rng(seed+7);splits=[[],[],[]]
    for cls in [0,1]:
        ix=r.permutation(np.flatnonzero(y==cls));cut1=round(.6*len(ix));cut2=round(.8*len(ix))
        for dst,z in zip(splits,[ix[:cut1],ix[cut1:cut2],ix[cut2:]]):dst.extend(z)
    b=np.random.default_rng(seed+9000).uniform(.05,.30,20);b[:6]=.4;delta=np.r_[np.full(6,.4),np.zeros(14)]
    return X,y,u,intercept,[np.sort(s) for s in splits],b,delta

def verify_seed(seed,smoke=False):
    root=HERE/('smoke_results' if smoke else 'run_results')/f'seed{seed}'
    result=js(root/'result.json');manifest=js(root/'manifest.json')
    assert result['status']=='complete' and result['data_seed']==seed and result['fits_completed']==20
    assert sha(root/'manifest.json')==result['manifest_sha256']
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}-{'manifest.json','result.json'}
    assert actual==set(manifest),('file_set',seed,actual^set(manifest))
    for name,info in manifest.items():
        p=root/name;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],p;COUNTS['file_hashes']+=1
    assert not (root/'failure.json').exists()
    d=np.load(root/'data.npz');ev=np.load(root/'evaluation_masks.npz');pre=np.load(root/'preprocessing.npz')
    X,y,u,intercept,(tr,va,te),b,delta=independent_generation(seed)
    for k,a in [('X',X),('y',y),('U_label',u),('train_rows',tr),('val_rows',va),('test_rows',te),('b',b),('delta',delta)]:eq(d[k],a,(seed,k))
    eq(np.sort(np.r_[tr,va,te]),np.arange(8000),'partition')
    assert result['data_meta']['intercept']==intercept and result['data_meta']['positives']==int(y.sum())
    U={}
    for name,offset,rows in [('train',1,tr),('val',2,va),('test',3,te)]:U[name]=np.random.default_rng(seed*10+offset).random((len(rows),20));eq(d['U_'+name],U[name],name)
    for s in [9107,9108,9109]:eq(d[f'U_mask_{s}'],np.random.default_rng(s).random((len(tr),20)),f'maskU{s}')
    for s in [101,202,303]:eq(d[f'U_stress_{s}'],np.random.default_rng(s).random((len(va),20)),f'stressU{s}')
    def natural(rows,m,uniform,key,metakey):
        yy=y[rows];pi=yy.mean();q1=b+(1-pi)*delta*m;q0=b-pi*delta*m
        mask=uniform<np.where(yy[:,None]==1,q1,q0);eq(ev[key],mask,key)
        z=X[rows].copy();z[mask]=np.nan
        meta=result[metakey] if metakey!='target_missing' else result[metakey][f'gap_{m:g}']
        close(meta['q1'],q1,key+'q1');close(meta['q0'],q0,key+'q0');close([meta['expected_overall_missing']],[b.mean()],key+'expect')
        close([meta['actual_missing']],[mask.mean()],key+'actual')
        return z,mask
    train,tm=natural(tr,1,U['train'],'train_natural','train_missing')
    val,vm=natural(va,1,U['val'],'val_natural','validation_missing')
    sets={'val':{'S0':val},'test':{}}
    for s in [101,202,303]:
        new=(~vm)&(d[f'U_stress_{s}']<.1);eq(ev[f'val_S1_10_{s}'],new,'stressmask');z=val.copy();z[new]=np.nan;sets['val'][f'S1_10_{s}']=z
    for m in MS:sets['test'][f'gap_{m:g}']=natural(te,m,U['test'],f'target_gap_{m:g}','target_missing')[0]
    median=np.nanmedian(train,axis=0);median=np.where(np.isnan(median),0.,median);eq(pre['medians'],median,'median')
    cnt=np.bincount(y[tr],minlength=2);miss=np.array([tm[y[tr]==c].sum(0) for c in [0,1]]);q=miss/cnt[:,None]
    for k,a in [('q',q),('counts',cnt),('missing_counts',miss)]:eq(pre[k],a,k)
    qmax=q.max(0);qbar=miss.sum(0)/len(tr);visible=~tm
    records=js(root/'fit_records.json');assert [r['tag'] for r in records]==TAGS
    assert {p.stem for p in (root/'models').glob('*.joblib')}==set(TAGS)
    assert {p.stem for p in (root/'masks').glob('*.npz')}==set(TAGS)
    assert {p.stem for p in (root/'predictions').glob('*.npz')}=={f'{tag}__{part}' for tag in TAGS for part in ['val','test']}
    cfg=js(HERE/'config.json');single={'val':{},'test':{}}
    for tag,record in zip(TAGS,records):
        model_info=joblib.load(root/'models'/f'{tag}.joblib');model=model_info['model'];mask=np.load(root/'masks'/f'{tag}.npz')
        if tag in ('M3','M4'):alpha=0.;new=np.zeros_like(tm);expect=np.zeros(20);blind=np.zeros(20);seedmask=None;arm=tag
        else:
            arm=tag[0];alpha=int(tag[1:4])/100;seedmask=int(tag.split('seed')[1]);uniform=d[f'U_mask_{seedmask}']
            expect=alpha*(cnt[:,None]*(qmax-q)).sum(0);blind=np.divide(alpha*(qmax-qbar),1-qbar,out=np.zeros(20),where=qbar<1)
            conditional=np.divide(alpha*(qmax-q),1-q,out=np.zeros_like(q),where=q<1)
            new=visible&(uniform<(conditional[y[tr]] if arm=='C' else blind))
            close(expect,blind*visible.sum(0),'budget_equal',1e-8)
        for k,a in [('new_mask',new),('row_deleted',new.sum(1)),('col_deleted',new.sum(0)),('class_col_deleted',np.array([new[y[tr]==c].sum(0) for c in [0,1]]))]:eq(mask[k],a,(tag,k))
        close(mask['expected_per_col'],expect,'expected_budget',1e-8);close(mask['blind_probabilities'],blind,'blindprob')
        assert not (new&tm).any();COUNTS['training_masks']+=1
        assert model_info['data_seed']==seed and model_info['tag']==tag and model_info['alpha']==alpha and model_info['arm']==arm and model_info['mask_seed']==seedmask
        assert model_info['config_sha256']==sha(HERE/'config.json');eq(model_info['train_rows'],tr,'model train rows');eq(model_info['medians'],median,'model median')
        assert model.get_params()==cfg['model_params']==record['params'];assert model.n_iter_==record['n_iter'];assert record['status']=='success';assert record['model_sha256']==sha(root/'models'/f'{tag}.joblib')
        assert model.n_features_in_==(20 if tag=='M4' else 40)
        COUNTS['models_reloaded']+=1
        for part,rows in [('val',va),('test',te)]:
            saved=np.load(root/'predictions'/f'{tag}__{part}.npz');eq(saved['row_indices'],rows,'prediction rows');eq(saved['y'],y[rows],'prediction y')
            assert set(saved.files)=={'row_indices','y',*sets[part]}
            single[part][tag]={}
            for condition,xx in sets[part].items():
                pred=model.predict_proba(reload_input(xx,median,tag))[:,1]
                err=close(pred,saved[condition],(tag,part,condition));ERRORS['prediction_max_abs']=max(ERRORS['prediction_max_abs'],err)
                single[part][tag][condition]=pred;COUNTS['single_prediction_arrays']+=1
    ens=np.load(root/'ensemble_predictions.npz');metrics=js(root/'metrics.json');selection=js(root/'selection.json');ens_calc={'val':{},'test':{}}
    for part in ['val','test']:
        ens_calc[part]={tag:single[part][tag] for tag in ['M3','M4']}
        for tag in BASE_CONFIGS[2:]:ens_calc[part][tag]={k:np.mean([single[part][f'{tag}_seed{s}'][k] for s in [9107,9108,9109]],axis=0) for k in sets[part]}
    for arm in ['C','B']:
        candidates=[];base=ap(y[va],ens_calc['val']['M3']['S0'])
        for alpha in [0,.25,.5,1]:
            tag='M3' if alpha==0 else f'{arm}{int(alpha*100):03d}';scores=[ap(y[va],ens_calc['val'][tag][k]) for k in VALS]
            candidates.append({'alpha':alpha,'config':tag,'s0':scores[0],'worst4':min(scores),'eligible':scores[0]>=base-.01})
        for saved,calc in zip(selection[arm]['candidates'],candidates):
            assert saved['alpha']==calc['alpha'] and saved['config']==calc['config'] and saved['eligible']==calc['eligible'];close([saved['s0'],saved['worst4']],[calc['s0'],calc['worst4']],'selection scores')
        chosen=sorted([x for x in candidates if x['eligible']],key=lambda x:(-x['worst4'],x['alpha']))[0]
        assert selection[arm]['chosen']['config']==chosen['config'];assert result['source_selection'][arm]==selection[arm]
        for part in ['val','test']:ens_calc[part][arm+'_selected']=ens_calc[part][chosen['config']]
        COUNTS['selections']+=1
    recomputed={}
    for tag in CONFIGS:
        recomputed[tag]={}
        for part,rows in [('val',va),('test',te)]:
            recomputed[tag][part]={}
            for cond,pred in ens_calc[part][tag].items():
                err=close(ens[f'{tag}__{part}__{cond}'],pred,'ensemble');ERRORS['ensemble_max_abs']=max(ERRORS['ensemble_max_abs'],err);COUNTS['ensemble_prediction_arrays']+=1
                score=ap(y[rows],pred);err=close([score],[metrics[tag][part][cond]],'AP');ERRORS['ap_max_abs']=max(ERRORS['ap_max_abs'],err);COUNTS['ap_values']+=1
                recomputed[tag][part][cond]=score
                if part=='test':close([score],[result['metrics'][tag][cond]],'result AP')
    for key,a in [('y_val',y[va]),('y_test',y[te]),('train_rows',tr),('val_rows',va),('test_rows',te)]:eq(ens[key],a,key)
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()];names=[e['event'] for e in events]
    assert names.count('selection_saved')==1 and names.index('selection_saved')<names.index('target_prediction_started')
    assert names.count('fit_completed')==20 and max(i for i,n in enumerate(names) if n=='fit_completed')<names.index('selection_saved')
    assert events[names.index('selection_saved')]['selection_sha256']==sha(root/'selection.json')
    assert names[-1]=='complete' and 'failed' not in names
    COUNTS['seeds']+=1;print(json.dumps({'verified_seed':seed,**COUNTS}),flush=True)
    return {'data_seed':seed,'metrics':{c:recomputed[c]['test'] for c in CONFIGS},'selection':selection,'result':result}

def verify_statistics(results):
    summary=js(HERE/'summary.json');assert summary['status']=='complete' and summary['fits_completed']==600
    assert [s['data_seed'] for s in summary['seeds']]==list(range(601,631))
    for saved,checked in zip(summary['seeds'],results):assert saved==checked['result'],'summary seed mismatch'
    ana=HERE/'analysis';report=js(ana/'statistical_summary.json');draw=np.load(ana/'bootstrap_seed_indices.npz')
    fixed=np.array([[r['metrics']['C025'][k]-r['metrics']['B025'][k] for k in GAPS] for r in results])
    selected=np.array([[r['metrics']['C_selected'][k]-r['metrics']['B_selected'][k] for k in GAPS] for r in results])
    vectors=np.column_stack([fixed.mean(1),fixed,selected]);ix=np.random.default_rng(20261108).integers(0,30,size=(20000,30));eq(draw['indices'],ix,'bootstrap indices')
    boot=np.array([vectors[row].mean(0) for row in ix]);close(draw['bootstrap_means'],boot,'bootstrap means')
    interval=np.quantile(boot,[.025,.975],axis=0);assert report['input_sha256']==sha(HERE/'summary.json');assert report['analysis_script_sha256']==sha(HERE/'analyze_results.py')
    def checkrow(row,x,col,test=False):
        n=len(x);mean=x.mean();sd=x.std(ddof=1);half=stats.t.ppf(.975,n-1)*sd/math.sqrt(n)
        expected=[mean,sd,np.median(x),x.min(),x.max(),mean-half,mean+half,half,interval[0,col],interval[1,col]]
        keys=['mean','sd','median','min','max','ci95_t_low','ci95_t_high','ci95_t_halfwidth','bootstrap_percentile_low','bootstrap_percentile_high']
        close([row[k] for k in keys],expected,'stats summary');assert row['n']==30
        if test:
            tt=stats.ttest_1samp(x,0);close([row['t'],row['p_nominal'],row['paired_dz']],[tt.statistic,tt.pvalue,mean/sd],'test');assert row['df']==29
    checkrow(report['primary'],vectors[:,0],0)
    assert report['primary']['precision_halfwidth_target']==.005
    assert report['primary']['precision_met']==bool(report['primary']['ci95_t_halfwidth']<=.005)
    ps=[]
    for j,row in enumerate(report['secondary_comparisons']):
        assert row['comparison']==('fixed_C025_minus_B025' if j<9 else 'selected_C_minus_B') and row['m']==MS[j%9]
        checkrow(row,vectors[:,j+1],j+1,True);ps.append(stats.ttest_1samp(vectors[:,j+1],0).pvalue)
    order=sorted(range(18),key=lambda i:ps[i]);adj=[0.]*18;running=0.
    for rank,i in enumerate(order):running=max(running,(18-rank)*ps[i]);adj[i]=min(1.,running)
    close([r['p_holm18'] for r in report['secondary_comparisons']],adj,'Holm18')
    assert [r['reject_holm18_05'] for r in report['secondary_comparisons']]==[p<.05 for p in adj]
    return {'independent_ap_inputs':True,'all_18_comparisons':True,'shared_bootstrap_draws':20000,'primary_and_secondary_intervals':True,'holm_family_size':18}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--skip-analysis',action='store_true');args=parser.parse_args()
    target=HERE/'audit'/('smoke_independent_verification.json' if args.smoke else 'independent_verification.json')
    try:
        with threadpool_limits(2):
            cfg=js(HERE/'config.json')
            if not args.smoke:
                freeze=js(HERE/'freeze.json')
                for name,digest in freeze['files'].items():assert sha(HERE/name)==digest,name
            seeds=[cfg['smoke_seed']] if args.smoke else list(range(601,631))
            results=[verify_seed(seed,args.smoke) for seed in seeds]
            stat={'status':'not_requested'} if args.smoke or args.skip_analysis else verify_statistics(results)
        write(target,{'status':'passed','scope':'interface smoke' if args.smoke else '30 new formal seeds','counts':COUNTS,'errors':ERRORS,
                      'statistics':stat,'verifier_sha256':sha(__file__),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                      'limit':'Artifacts and arithmetic checked by a separate assistant in the same session; no historical provenance proof or independent peer review'})
    except BaseException as exc:
        write(target,{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'counts':COUNTS,'errors':ERRORS,'verifier_sha256':sha(__file__)})
        raise
if __name__=='__main__':main()
