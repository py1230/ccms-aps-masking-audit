"""Paired information channels and repeated training randomness; retrospective APS."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import argparse, hashlib, json, platform, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits, threadpool_info
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'));sys.path.insert(0,str(HERE/'historical'))
from aps_io import load_aps_csv,split_features
from pipeline import MedianImputer
from ccms import estimate_class_rates,class_blind_rates,mask_class_conditional,mask_class_blind
from run_cb_firstlook import build_eval_inputs
SEL=['S0','S1_10_101','S1_10_202','S1_10_303']
GROUPS=['S0','S1_10','S1_30','S2','S3']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def environment():
    return {'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
            'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
            'scipy':scipy.__version__,'joblib':joblib.__version__,'threadpool_info':threadpool_info()}
def matrix(x,med,channel):
    if channel=='MO':return np.isnan(x).astype(float)
    filled=np.where(np.isnan(x),med[None,:],x)
    if channel=='M2':return filled
    if channel=='M3':return np.hstack([filled,np.isnan(x).astype(float)])
    raise ValueError(channel)
def get_mask(x,y,arm,alpha,seed):
    if arm=='BASE':return np.zeros(x.shape,dtype=bool)
    if arm=='C':return mask_class_conditional(x,y,alpha,seed=seed)[1]['new_mask']
    if arm=='B':
        q,n,_=estimate_class_rates(x,y)
        return mask_class_blind(x,class_blind_rates(q,n,alpha),seed=seed)[1]['new_mask']
    raise ValueError(arm)
def specs(cfg,rep):
    ans=[('BASE','BASE',0.,None)]
    for alpha in cfg['alphas'][1:]:
        for arm in ['C','B']:
            for seed in rep['mask_seeds']:ans.append((f'{arm}{round(alpha*100):03d}_seed{seed}',arm,alpha,seed))
    return ans

def check(cfg,freeze=True):
    if freeze:
        fr=json.loads((HERE/'freeze.json').read_text())
        for name,row in fr['files'].items():
            p=Path(row['path']);p=p if p.is_absolute() else HERE/p
            assert sha(p)==row['sha256'],f'frozen file changed: {name}'
    for key,value in cfg['input_environment'].items():assert environment()[key]==value,(key,environment()[key],value)
    d,_=load_aps_csv(cfg['train_csv']);dt,_=load_aps_csv(cfg['test_csv'])
    assert list(d.columns)==list(dt.columns)
    x,y=split_features(d);xt,yt=split_features(dt)
    assert x.shape==(60000,170) and xt.shape==(16000,170)
    assert y.sum()==1000 and yt.sum()==375
    assert np.isfinite(x[~np.isnan(x)]).all() and np.isfinite(xt[~np.isnan(xt)]).all()
    assert sha(HERE/'split_indices.npz')==cfg['split_source_sha256']
    with np.load(HERE/'split_indices.npz',allow_pickle=False) as z:ix={k:z[k] for k in z.files}
    for split in cfg['split_seeds']:
        tr,va,rs=[ix[f's{split}_{k}'] for k in ['train','val','reserve']]
        assert [len(tr),len(va),len(rs)]==[42000,9000,9000]
        assert np.array_equal(np.sort(np.r_[tr,va,rs]),np.arange(len(x)))
        assert [int(y[a].sum()) for a in [tr,va,rs]]==[700,150,150]
    return x,y,xt,yt,ix,np.asarray([c for c in d.columns if c!='class'])

def choose(metrics,cfg):
    chosen={};candidates={}
    for arm in ['C','B']:
        rows=[]
        for a in cfg['alphas']:
            name='BASE' if a==0 else f'{arm}{round(a*100):03d}'
            s0=metrics[name]['val__S0']
            rows.append({'alpha':a,'config':name,'s0':s0,'worst4':min(metrics[name]['val__'+c] for c in SEL),
                         'eligible':bool(s0>=metrics['BASE']['val__S0']-cfg['s0_tolerance'])})
        chosen[arm]=max([r for r in rows if r['eligible']],key=lambda r:(r['worst4'],-r['alpha']))
        candidates[arm]=rows
    return {'chosen':chosen,'candidates':candidates,'selected_before_test_prediction':True,'timestamp_utc':stamp()}

def smoke(cfg):
    x,y,xt,yt,ix,features=check(cfg,freeze=False)
    out=HERE/'smoke_results';out.mkdir(exist_ok=False)
    tr=ix['s11_train'];tr=np.r_[tr[y[tr]==0][:1960],tr[y[tr]==1][:40]]
    xx,yy=x[tr],y[tr];med=MedianImputer().fit(xx).medians_
    mask=get_mask(xx,yy,'C',.25,19107);xm=xx.copy();xm[mask]=np.nan
    errs={}
    for channel in cfg['channels']:
        m=HistGradientBoostingClassifier(**cfg['model_params']).fit(matrix(xm,med,channel),yy)
        joblib.dump({'model':m,'medians':med,'channel':channel},out/f'{channel}.joblib')
        m2=joblib.load(out/f'{channel}.joblib')['model']
        a=m.predict_proba(matrix(xt[:300],med,channel))[:,1];b=m2.predict_proba(matrix(xt[:300],med,channel))[:,1]
        errs[channel]=float(np.max(np.abs(a-b)));assert errs[channel]==0
    assert matrix(xm,med,'M2').shape[1]==170 and matrix(xm,med,'M3').shape[1]==340
    assert np.array_equal(matrix(xm,med,'MO'),matrix(xm,med,'M3')[:,170:])
    assert np.array_equal(matrix(xm,med,'M2'),matrix(xm,med,'M3')[:,:170])
    dump(out/'smoke.json',{'status':'passed','train_rows':len(tr),'test_rows':300,'reload_error':errs,
                          'environment':environment(),'timestamp_utc':stamp(),'scope':'interface only, no AP used to adapt design'})
    print('SMOKE PASSED',flush=True)

def run(cfg,channel):
    x,y,xt,yt,ix,features=check(cfg)
    parent=HERE/'run_results';parent.mkdir(exist_ok=True)
    out=parent/channel;out.mkdir(exist_ok=False)
    start=time.perf_counter();fit_total=0
    dump(out/'environment.json',{**environment(),'started_utc':stamp(),'freeze_sha256':sha(HERE/'freeze.json'),'channel':channel})
    def log(obj):
        obj={'time_utc':stamp(),'channel':channel,**obj}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        print(json.dumps(obj,ensure_ascii=False),flush=True)
    log({'event':'start','expected_fits':cfg['expected_models_per_channel']})
    test_evals={k:v[0] for k,v in build_eval_inputs(xt).items()}
    np.savez_compressed(out/'test_evaluation_masks.npz',**{k:np.isnan(v)&~np.isnan(xt) for k,v in test_evals.items()})
    summaries=[]
    try:
        for rep in cfg['repeats']:
            (out/f'rep{rep["id"]}').mkdir()
            params={**cfg['model_params'],'random_state':rep['model_seed']}
            mspecs=specs(cfg,rep)
            for split in cfg['split_seeds']:
                dest=out/f'rep{rep["id"]}'/f'split{split}';dest.mkdir()
                for folder in ['models','training_masks','diagnostics','predictions']:(dest/folder).mkdir()
                tr,va=ix[f's{split}_train'],ix[f's{split}_val'];xx,yy=x[tr],y[tr]
                med=MedianImputer().fit(xx).medians_;q,counts,miss=estimate_class_rates(xx,yy)
                np.savez_compressed(dest/'preprocessing.npz',medians=med,feature_names=features,train_rows=tr,
                                    q=q,class_counts=counts,missing_counts=miss)
                evals={k:v[0] for k,v in build_eval_inputs(x[va]).items()}
                np.savez_compressed(dest/'evaluation_masks.npz',**{k:np.isnan(v)&~np.isnan(x[va]) for k,v in evals.items()})
                preds={};fits=[]
                for tag,arm,alpha,seed in mspecs:
                    mask=get_mask(xx,yy,arm,alpha,seed);assert not (mask&np.isnan(xx)).any()
                    xm=xx.copy();xm[mask]=np.nan
                    expected=alpha*(counts[:,None]*(q.max(0)[None,:]-q)).sum(0)
                    eb=class_blind_rates(q,counts,alpha)*(~np.isnan(xx)).sum(0)
                    assert np.allclose(expected,eb,rtol=0,atol=cfg['tolerances']['budget_atol'])
                    np.savez_compressed(dest/'training_masks'/f'{tag}.npz',new_mask=mask)
                    np.savez_compressed(dest/'diagnostics'/f'{tag}.npz',row_deleted=mask.sum(1),col_deleted=mask.sum(0),
                                        class_col_deleted=np.stack([mask[yy==c].sum(0) for c in [0,1]]),
                                        expected_col=expected,q_after=estimate_class_rates(xm,yy)[0])
                    t=time.perf_counter();model=HistGradientBoostingClassifier(**params)
                    model.fit(matrix(xm,med,channel),yy);fit_seconds=time.perf_counter()-t;fit_total+=1
                    modelpath=dest/'models'/f'{tag}.joblib'
                    joblib.dump({'model':model,'medians':med,'train_rows':tr,'arm':arm,'alpha':alpha,'mask_seed':seed,
                                 'model_seed':rep['model_seed'],'repeat':rep['id'],'channel':channel,
                                 'config_sha256':sha(HERE/'config.json')},modelpath,compress=3)
                    ps={k:model.predict_proba(matrix(v,med,channel))[:,1] for k,v in evals.items()}
                    for p in ps.values():assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                    np.savez_compressed(dest/'predictions'/f'{tag}__val.npz',row_indices=va,y=y[va],**ps);preds[tag]=ps
                    rec={'tag':tag,'n_iter':int(model.n_iter_),'do_early_stopping':bool(model.do_early_stopping_),
                         'fit_seconds':fit_seconds,'model_sha256':sha(modelpath),'effective_params':model.get_params()}
                    fits.append(rec)
                    log({'event':'fit','repeat':rep['id'],'split':split,'fits_completed':fit_total,
                         'elapsed_seconds':time.perf_counter()-start,'tag':tag,'fit_seconds':fit_seconds,'n_iter':int(model.n_iter_)})
                    del model,xm,mask
                dump(dest/'fit_records.json',fits)
                configs=list(dict.fromkeys(t.split('_seed')[0] for t,_,_,_ in mspecs))
                ensemble={'train_rows':tr,'val_rows':va,'test_rows':np.arange(len(yt)),'y_val':y[va],'y_test':yt}
                metrics={c:{} for c in configs}
                def aggregate(ps,dataset,labels):
                    for c in configs:
                        tags=[t for t,_,_,_ in mspecs if t.split('_seed')[0]==c]
                        for cond in next(iter(ps.values())):
                            p=np.mean([ps[t][cond] for t in tags],axis=0)
                            ensemble[f'{c}__{dataset}__{cond}']=p
                            metrics[c][f'{dataset}__{cond}']=float(average_precision_score(labels,p))
                aggregate(preds,'val',y[va]);selection=choose(metrics,cfg);dump(dest/'selection.json',selection)
                log({'event':'selection_frozen','repeat':rep['id'],'split':split,
                     'chosen':{k:r['alpha'] for k,r in selection['chosen'].items()},'selection_sha256':sha(dest/'selection.json')})
                del evals,preds
                preds={}
                for tag,_,_,_ in mspecs:
                    model=joblib.load(dest/'models'/f'{tag}.joblib')['model']
                    ps={k:model.predict_proba(matrix(v,med,channel))[:,1] for k,v in test_evals.items()}
                    for p in ps.values():assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                    np.savez_compressed(dest/'predictions'/f'{tag}__test.npz',row_indices=np.arange(len(yt)),y=yt,**ps);preds[tag]=ps
                aggregate(preds,'test',yt)
                np.savez_compressed(dest/'ensemble_predictions.npz',**ensemble);dump(dest/'metrics.json',metrics)
                grouped={}
                for c,values in metrics.items():
                    grouped[c]={ds:{cond:values[f'{ds}__S0'] if cond=='S0' else
                                         float(np.mean([v for k,v in values.items() if k.startswith(f'{ds}__{cond}_')]))
                                    for cond in GROUPS} for ds in ['val','test']}
                for arm in ['C','B']:grouped[arm+'_selected']=grouped[selection['chosen'][arm]['config']]
                summaries.append({'repeat':rep['id'],'model_seed':rep['model_seed'],'split':split,
                                  'chosen_alphas':{k:r['alpha'] for k,r in selection['chosen'].items()},'ap':grouped})
                dump(out/'summary.json',{'status':'running','channel':channel,'fits_completed':fit_total,'cells':summaries})
                log({'event':'cell_complete','repeat':rep['id'],'split':split,'fits_completed':fit_total,
                     'elapsed_seconds':time.perf_counter()-start})
                del preds,ensemble
        assert fit_total==cfg['expected_models_per_channel']
        dump(out/'summary.json',{'status':'complete','channel':channel,'fits_completed':fit_total,'cells':summaries,
                                'finished_utc':stamp(),'wall_seconds':time.perf_counter()-start,
                                'inference':'conditional training randomness, fixed rows and eval masks; descriptive only'})
        log({'event':'complete','fits_completed':fit_total,'wall_seconds':time.perf_counter()-start})
    except BaseException as e:
        dump(out/'failure.json',{'timestamp_utc':stamp(),'fits_completed':fit_total,'error':repr(e),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--channel',choices=['M2','M3','MO']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    with threadpool_limits(limits=cfg['threads']):
        if args.check:
            x,y,xt,yt,ix,features=check(cfg);print(json.dumps({'status':'passed','expected_models':cfg['expected_models'],'environment':environment()}))
        elif args.smoke:smoke(cfg)
        elif args.channel:run(cfg,args.channel)
        else:parser.error('Specify --channel, --check, or --smoke')
