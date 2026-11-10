"""APS stratified training-ID bootstrap; fixed external validation and test."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import argparse,hashlib,inspect,json,platform,resource,shutil,sys,time,traceback,warnings
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits,threadpool_info
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from aps_io import load_aps_csv,split_features
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
SEL=['S0','S1_10_101','S1_10_202','S1_10_303']
CONDITIONS=SEL+[f'S1_30_{s}' for s in [101,202,303]]+[f'S2_{s}' for s in [1101,1202,1303]]+[f'S3_{s}' for s in [2101,2202,2303]]
GROUPS=['S0','S1_10','S1_30','S2','S3']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def config():return json.loads((HERE/'config.json').read_text())
def environment():return {'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'executable':sys.executable,'platform':platform.platform(),'threadpools':threadpool_info()}
def data():
    d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');x,y=split_features(d)
    dt,_=load_aps_csv(HERE/'raw_data/aps_failure_test_set.csv');xt,yt=split_features(dt)
    assert x.shape==(60000,170) and xt.shape==(16000,170) and y.sum()==1000 and yt.sum()==375
    assert list(d.columns)==list(dt.columns)
    return x,y,xt,yt,np.array([c for c in d.columns if c!='class'])
def fitmed(x):
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x,axis=0)
    return np.where(np.isnan(med),0.,med)
def matrix(x,med):return np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])
def evalmask(x,cond):
    visible=~np.isnan(x);out=np.zeros(x.shape,dtype=bool)
    if cond=='S0':return out
    p=cond.split('_');rng=np.random.default_rng(int(p[-1]))
    if p[0]=='S1':return visible&(rng.random(x.shape)<int(p[1])/100)
    if p[0]=='S2':
        rows=rng.choice(len(x),size=round(.5*len(x)),replace=False);out[rows]=visible[rows]&(rng.random((len(rows),x.shape[1]))<.6)
    elif p[0]=='S3':
        cols=rng.choice(x.shape[1],size=max(1,round(.3*x.shape[1])),replace=False);out[:,cols]=visible[:,cols]
    else:raise ValueError(cond)
    return out
def specs(cfg):
    return [('BASE','BASE',0.,None)]+[(f'{arm}{round(alpha*100):03d}_seed{seed}',arm,alpha,seed) for alpha in cfg['alphas'][1:] for arm in ['C','B'] for seed in cfg['mask_seeds']]
def maskfor(x,y,arm,alpha,seed):
    if arm=='BASE':return np.zeros(x.shape,dtype=bool)
    if arm=='C':return mask_class_conditional(x,y,alpha,seed=seed)[1]['new_mask']
    q,n,_=estimate_class_rates(x,y);return mask_class_blind(x,class_blind_rates(q,n,alpha),seed=seed)[1]['new_mask']
def choose(metrics,cfg):
    chosen={};candidates={}
    for arm in ['C','B']:
        rows=[]
        for alpha in cfg['alphas']:
            name='BASE' if alpha==0 else f'{arm}{round(alpha*100):03d}'
            rows.append({'alpha':alpha,'config':name,'s0':metrics[name]['val__S0'],'worst4':min(metrics[name]['val__'+c] for c in SEL),'eligible':bool(metrics[name]['val__S0']>=metrics['BASE']['val__S0']-cfg['s0_tolerance'])})
        chosen[arm]=max([r for r in rows if r['eligible']],key=lambda r:(r['worst4'],-r['alpha']));candidates[arm]=rows
    return {'chosen':chosen,'candidates':candidates,'selected_before_test_prediction':True,'timestamp':stamp()}
def prepare():
    assert not (HERE/'config.json').exists()
    oldroot=HERE.parent/'aps_channels_repeats_20261108';old=json.loads((oldroot/'config.json').read_text())
    cfg={'design':'stratified original-training-ID bootstrap, fixed validation/test, no internal early stopping',
        'split':11,'bootstrap_seeds':list(range(90101,90121)),'alphas':[0,.25,.5,1.],'mask_seeds':[19107,19108,19109],
        'model_params':{**old['model_params'],'early_stopping':False,'max_iter':100,'random_state':2026},
        'input_environment':old['input_environment'],'threads':2,'max_unit_seconds':7200,'expected_fits':399,
        'fits_per_unit':19,'bootstrap_repeats':20,'anchor_separate':True,'s0_tolerance':.01,
        'train_class_counts':[41300,700],'val_class_counts':[8850,150],'test_class_counts':[15625,375],
        'bootstrap_algorithm':'rng.default_rng(seed); choice(train_rows[y==0],41300,replace=True); choice(train_rows[y==1],700,replace=True); concatenate in class order; rng.permutation(concatenation)',
        'primary':'selected_C minus selected_B, official test S1_30; descriptive bootstrap mean/SD/min/max',
        'key_secondary':'fixed C025 minus B025, test S1_30',
        'statistics':'20 empirical training-bootstrap repeats only; descriptive mean/sampleSD/min/max, no p/CI; anchor excluded from repeats',
        'scope':'fixed empirical split11, class counts, external validation/test, model/mask seeds; no population, cluster, test-sample or deployment inference',
        'refit_after_selection':False,'selection_conditions':SEL,'test_conditions':CONDITIONS,
        'array_hash':'SHA256(dtype.str UTF8 + compact JSON list(shape) UTF8 + C order bytes)',
        'scientific_stop':'all 399 formal fits and independent verification; no effect-adaptive expansion'}
    dump(HERE/'config.json',cfg);(HERE/'raw_data').mkdir()
    sources={}
    for key in ['train_csv','test_csv']:
        src=Path(old[key]);dest=HERE/'raw_data'/src.name;shutil.copy2(src,dest);sources[str(dest.relative_to(HERE))]={'source':str(src),'sha256':sha(src)}
    sources['split_indices.npz']={'source':str(oldroot/'split_indices.npz'),'sha256':sha(HERE/'split_indices.npz')}
    dump(HERE/'input_sources.json',sources)
    x,y,xt,yt,features=data();ix=np.load(HERE/'split_indices.npz');tr=ix['s11_train'];va=ix['s11_val'];reserve=ix['s11_reserve']
    assert len(tr)==42000 and len(va)==9000 and len(reserve)==9000 and not np.intersect1d(tr,va).size
    for rows,counts in [(tr,cfg['train_class_counts']),(va,cfg['val_class_counts'])]:assert np.bincount(y[rows],minlength=2).tolist()==counts
    idx=HERE/'prepared_indices';idx.mkdir()
    for seed in [None]+cfg['bootstrap_seeds']:
        name='anchor' if seed is None else f'seed{seed}';rng=np.random.default_rng(seed) if seed is not None else None
        states={}
        if seed is None:rows=tr.copy()
        else:
            states['before']=rng.bit_generator.state
            byclass=[rng.choice(tr[y[tr]==c],size=cfg['train_class_counts'][c],replace=True) for c in [0,1]]
            states['before_permutation']=rng.bit_generator.state;rows=rng.permutation(np.concatenate(byclass));states['after']=rng.bit_generator.state
        counts=np.bincount(rows,minlength=60000);assert not np.intersect1d(rows,va).size and not np.intersect1d(rows,reserve).size
        np.savez_compressed(idx/f'{name}.npz',train_rows=rows,multiplicity=counts,base_train_rows=tr,val_rows=va,reserve_rows=reserve)
        dump(idx/f'{name}.json',{'unit':name,'seed':seed,'rng_states':states,'unique_rows':int(np.count_nonzero(counts)),'duplicate_occurrences':int(len(rows)-np.count_nonzero(counts)),'class_counts':np.bincount(y[rows],minlength=2).tolist(),'rows_sha256':arrsha(rows),'multiplicity_sha256':arrsha(counts)})
    np.savez_compressed(HERE/'validation_masks.npz',**{c:evalmask(x[va],c) for c in SEL})
    np.savez_compressed(HERE/'test_masks.npz',**{c:evalmask(xt,c) for c in CONDITIONS})
    dump(HERE/'environment.json',environment());print('PREPARED',flush=True)
def smoke():
    cfg=config();out=HERE/'interface_smoke';out.mkdir(exist_ok=False);started=time.perf_counter()
    x,y,xt,yt,features=data();ix=np.load(HERE/'split_indices.npz');base=ix['s11_train'];va=ix['s11_val'];rng=np.random.default_rng(990001)
    tr=rng.permutation(np.r_[rng.choice(base[y[base]==0],1967,replace=True),rng.choice(base[y[base]==1],33,replace=True)])
    xx=x[tr];yy=y[tr];med=fitmed(xx);mask=maskfor(xx,yy,'C',.25,19107);xm=xx.copy();xm[mask]=np.nan;train=matrix(xm,med)
    model=HistGradientBoostingClassifier(**cfg['model_params']);start=time.perf_counter();model.fit(train,yy);fit_seconds=time.perf_counter()-start
    assert model.n_iter_==100 and not model.do_early_stopping_
    joblib.dump({'model':model,'medians':med,'train_rows':tr,'training_matrix_sha256':arrsha(train)},out/'model.joblib',compress=3)
    np.savez_compressed(out/'inputs.npz',train_rows=tr,medians=med,new_mask=mask,labels=yy)
    predtimes={};shapes={};hashes={};reloaded=joblib.load(out/'model.joblib')['model'];reload_errors={}
    for ds,xe,rows,conditions in [('val',x[va],va,SEL),('test',xt,np.arange(len(xt)),CONDITIONS)]:
        predictions={};ev=np.load(HERE/('validation_masks.npz' if ds=='val' else 'test_masks.npz'))
        for c in conditions:
            q=xe.copy();q[ev[c]]=np.nan;mat=matrix(q,med);t=time.perf_counter();p=model.predict_proba(mat)[:,1];predtimes[ds+'__'+c]=time.perf_counter()-t
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all();p2=reloaded.predict_proba(mat)[:,1];err=float(np.max(np.abs(p-p2)));assert err==0
            predictions[c]=p;shapes[ds+'__'+c]=list(mat.shape);hashes[ds+'__'+c]=arrsha(mat);reload_errors[ds+'__'+c]=err
        np.savez_compressed(out/f'{ds}_probabilities.npz',row_indices=rows,**predictions)
    res={'status':'passed','scope':'one 2000-row interface/resource model, no AP calculated; not a formal fit','train_rows':2000,'unique_original_rows':int(len(np.unique(tr))),'class_counts':[1967,33],
         'fit_seconds':fit_seconds,'iterations':int(model.n_iter_),'early_stopping':False,'prediction_seconds':predtimes,'prediction_seconds_total':sum(predtimes.values()),
         'prediction_arrays':17,'evaluation_shapes':shapes,'evaluation_matrix_sha256':hashes,'reload_max_error':max(reload_errors.values()),
         'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.perf_counter()-started,'environment':environment(),
         'HGB_fit_signature':str(inspect.signature(HistGradientBoostingClassifier.fit)),'timestamp':stamp(),'runner_sha256':sha(__file__)}
    dump(out/'smoke.json',res);print(json.dumps(res),flush=True)
def freeze():
    assert not (HERE/'freeze.json').exists()
    names={'run_resample.py','summarize_results.py','launch_runs.py','config.json','PROTOCOL.md','environment.json','input_sources.json','split_indices.npz','validation_masks.npz','test_masks.npz'}
    for name in names:assert (HERE/name).is_file(),name
    assert len(list((HERE/'prepared_indices').glob('*.npz')))==21
    files={}
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if p.is_file() and '__pycache__' not in p.parts and (str(rel) in names or rel.parts[0] in ['source','raw_data','prepared_indices']):files[str(rel)]=sha(p)
    dump(HERE/'freeze.json',{'timestamp':stamp(),'environment':environment(),'files':files,'scope':'science code, protocol, input data, bootstrap IDs, fixed evaluation masks before formal fitting'})
def check():
    fr=json.loads((HERE/'freeze.json').read_text())
    for name,value in fr['files'].items():assert sha(HERE/name)==value,name
    for k,v in config()['input_environment'].items():assert environment()[k]==v,(k,environment()[k],v)
def run(unit):
    check();cfg=config();assert unit in ['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']]
    dest=HERE/'units'/unit;dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter();fits=0
    def log(obj):
        obj={'timestamp':stamp(),'unit':unit,**obj}
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    try:
        x,y,xt,yt,features=data();idx=np.load(HERE/'prepared_indices'/f'{unit}.npz');tr=idx['train_rows'];va=idx['val_rows'];xx=x[tr];yy=y[tr]
        assert np.bincount(yy,minlength=2).tolist()==cfg['train_class_counts'] and not np.intersect1d(tr,va).size
        med=fitmed(xx);q,counts,missing=estimate_class_rates(xx,yy)
        np.savez_compressed(dest/'preprocessing.npz',medians=med,q=q,class_counts=counts,missing_counts=missing,train_rows=tr,val_rows=va,feature_names=features)
        for folder in ['models','training_masks','diagnostics','predictions']:(dest/folder).mkdir()
        dump(dest/'input_hashes.json',{'X_train_raw':arrsha(xx),'y_train':arrsha(yy),'train_rows':arrsha(tr),'X_val_raw':arrsha(x[va]),'y_val':arrsha(y[va]),'X_test_raw':arrsha(xt),'y_test':arrsha(yt),'median':arrsha(med),'q':arrsha(q),'config_sha256':sha(HERE/'config.json'),'freeze_sha256':sha(HERE/'freeze.json')})
        allspecs=specs(cfg);valmasks=np.load(HERE/'validation_masks.npz');testmasks=np.load(HERE/'test_masks.npz');preds={};records=[]
        evalhash={}
        for ds,xe,conds,masks in [('val',x[va],SEL,valmasks),('test',xt,CONDITIONS,testmasks)]:
            for c in conds:
                xe2=xe.copy();xe2[masks[c]]=np.nan;evalhash[ds+'__'+c]=arrsha(matrix(xe2,med))
        dump(dest/'evaluation_matrix_hashes.json',evalhash)
        for tag,arm,alpha,seed in allspecs:
            mask=maskfor(xx,yy,arm,alpha,seed);assert not (mask&np.isnan(xx)).any();xm=xx.copy();xm[mask]=np.nan;train=matrix(xm,med);inputsha=arrsha(train)
            np.savez_compressed(dest/'training_masks'/f'{tag}.npz',new_mask=mask)
            np.savez_compressed(dest/'diagnostics'/f'{tag}.npz',row_deleted=mask.sum(1),col_deleted=mask.sum(0),class_col_deleted=np.stack([mask[yy==c].sum(0) for c in [0,1]]))
            model=HistGradientBoostingClassifier(**cfg['model_params']);t=time.perf_counter();model.fit(train,yy);fit_seconds=time.perf_counter()-t;fits+=1
            assert model.n_iter_==100 and not model.do_early_stopping_
            path=dest/'models'/f'{tag}.joblib';joblib.dump({'model':model,'medians':med,'train_rows':tr,'unit':unit,'arm':arm,'alpha':alpha,'mask_seed':seed,'model_seed':2026,'channel':'M3','training_matrix_sha256':inputsha,'training_labels_sha256':arrsha(yy),'mask_sha256':arrsha(mask),'config_sha256':sha(HERE/'config.json')},path,compress=3)
            del train,xm
            p={}
            for c in SEL:
                z=x[va].copy();z[valmasks[c]]=np.nan;p[c]=model.predict_proba(matrix(z,med))[:,1]
                assert np.isfinite(p[c]).all() and (p[c]>=0).all() and (p[c]<=1).all()
            np.savez_compressed(dest/'predictions'/f'{tag}__val.npz',row_indices=va,y=y[va],**p);preds[tag]=p
            records.append({'tag':tag,'fit_seconds':fit_seconds,'n_iter':int(model.n_iter_),'do_early_stopping':bool(model.do_early_stopping_),'effective_params':model.get_params(),'model_sha256':sha(path),'training_matrix_sha256':inputsha,'mask_sha256':arrsha(mask)})
            dump(dest/'fit_records.json',records);log({'event':'fit','tag':tag,'fits':fits,'fit_seconds':fit_seconds,'elapsed_seconds':time.perf_counter()-start})
        names=list(dict.fromkeys(t.split('_seed')[0] for t,_,_,_ in allspecs));metrics={n:{} for n in names};ensembles={'y_val':y[va],'y_test':yt,'train_rows':tr,'val_rows':va,'test_rows':np.arange(len(yt))}
        def aggregate(predictions,ds,labels,conditions):
            for name in names:
                tags=[t for t,_,_,_ in allspecs if t.split('_seed')[0]==name]
                for c in conditions:
                    p=np.mean([predictions[t][c] for t in tags],axis=0);ensembles[f'{name}__{ds}__{c}']=p;metrics[name][ds+'__'+c]=float(average_precision_score(labels,p))
        aggregate(preds,'val',y[va],SEL);selection=choose(metrics,cfg);dump(dest/'selection.json',selection)
        log({'event':'selection_frozen','selection_sha256':sha(dest/'selection.json'),'chosen':{a:r['alpha'] for a,r in selection['chosen'].items()}})
        log({'event':'test_prediction_started','selection_sha256':sha(dest/'selection.json')})
        preds={}
        for tag,_,_,_ in allspecs:
            model=joblib.load(dest/'models'/f'{tag}.joblib')['model'];p={}
            for c in CONDITIONS:
                z=xt.copy();z[testmasks[c]]=np.nan;p[c]=model.predict_proba(matrix(z,med))[:,1]
                assert np.isfinite(p[c]).all() and (p[c]>=0).all() and (p[c]<=1).all()
            np.savez_compressed(dest/'predictions'/f'{tag}__test.npz',row_indices=np.arange(len(yt)),y=yt,**p);preds[tag]=p
            log({'event':'test_prediction_saved','tag':tag,'selection_sha256':sha(dest/'selection.json'),'prediction_sha256':sha(dest/'predictions'/f'{tag}__test.npz')})
        aggregate(preds,'test',yt,CONDITIONS);np.savez_compressed(dest/'ensemble_predictions.npz',**ensembles);dump(dest/'metrics.json',metrics)
        grouped={name:{g:float(v['test__S0'] if g=='S0' else np.mean([p for k,p in v.items() if k.startswith('test__'+g+'_')])) for g in GROUPS} for name,v in metrics.items()}
        for a in ['C','B']:grouped[a+'_selected']=grouped[selection['chosen'][a]['config']]
        contrasts={}
        for a,b in [('C_selected','B_selected'),('C_selected','BASE'),('B_selected','BASE'),('C025','B025'),('C025','BASE'),('B025','BASE')]:contrasts[a+'-'+b]={g:grouped[a][g]-grouped[b][g] for g in GROUPS}
        dump(dest/'summary.json',{'status':'complete','unit':unit,'fits':fits,'bootstrap_seed':None if unit=='anchor' else int(unit[4:]),'chosen_alphas':{a:r['alpha'] for a,r in selection['chosen'].items()},'ap':grouped,'contrasts':contrasts,'seconds':time.perf_counter()-start,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'environment':environment(),'finished':stamp()})
        log({'event':'complete','fits':fits,'elapsed_seconds':time.perf_counter()-start})
    except BaseException:
        dump(dest/'failure.json',{'timestamp':stamp(),'fits_completed':fits,'traceback':traceback.format_exc()});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--unit');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:prepare()
        elif a.smoke:smoke()
        elif a.freeze:freeze()
        elif a.check:check()
        elif a.unit:run(a.unit)
        else:p.error('choose action')
if __name__=='__main__':main()
