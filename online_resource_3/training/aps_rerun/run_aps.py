"""APS retrospective rerun. Explicit current defaults, archived indices, full evidence."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE / 'historical'))
from aps_io import load_aps_csv, split_features, dev_split
from pipeline import MedianImputer
from ccms import estimate_class_rates, class_blind_rates, mask_class_conditional, mask_class_blind
from run_cb_firstlook import build_eval_inputs

SEL = ['S0', 'S1_10_101', 'S1_10_202', 'S1_10_303']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def stamp(): return datetime.now(timezone.utc).isoformat()

def matrix(x, med):
    return np.hstack([np.where(np.isnan(x), med[None, :], x), np.isnan(x).astype(float)])

def count_mask(x, counts, seed):
    visible = ~np.isnan(x)
    out = np.zeros(x.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    for j, k in enumerate(counts):
        k = int(k)
        assert 0 <= k <= visible[:, j].sum()
        if k: out[rng.choice(np.flatnonzero(visible[:, j]), size=k, replace=False), j] = True
    assert np.array_equal(out.sum(0), counts)
    return out

def get_mask(x, y, arm, alpha, seed):
    if arm == 'M3': return np.zeros(x.shape, dtype=bool)
    q, counts, _ = estimate_class_rates(x, y)
    if arm in ['C', 'A3']:
        _, info = mask_class_conditional(x, y, alpha, seed=seed)
    else:
        _, info = mask_class_blind(x, class_blind_rates(q, counts, alpha), seed=seed)
    if arm in ['A3', 'A4']:
        return count_mask(x, info['deleted_per_col'], seed * 1000 + (7 if arm == 'A3' else 11))
    return info['new_mask']

def model_specs(cfg):
    specs = [('M3', 'M3', 0., None)]
    for a in cfg['alphas'][1:]:
        for arm in ['C', 'B']:
            for seed in cfg['train_mask_seeds']:
                specs.append((f'{arm}{round(a*100):03d}_seed{seed}', arm, a, seed))
    for arm in ['A3', 'A4']:
        for seed in cfg['train_mask_seeds']:
            specs.append((f'{arm}_seed{seed}', arm, .25, seed))
    return specs

def choose(metrics, cfg):
    selected, candidates = {}, {}
    for arm in ['C', 'B']:
        rows = []
        for a in cfg['alphas']:
            name = 'M3' if a == 0 else f'{arm}{round(a*100):03d}'
            s0 = metrics[name]['val__S0']
            rows.append({'alpha': a, 'config': name, 's0': s0,
                         'worst4': min(metrics[name]['val__'+c] for c in SEL),
                         'eligible': bool(s0 >= metrics['M3']['val__S0'] - cfg['s0_tolerance'])})
        selected[arm] = max([r for r in rows if r['eligible']], key=lambda r:(r['worst4'], -r['alpha']))
        candidates[arm] = rows
    return {'chosen': selected, 'candidates': candidates,
            'selected_before_test_prediction': True, 'timestamp_utc': stamp()}

def load(cfg, check_freeze=True):
    if check_freeze:
        fr = json.loads((HERE/'freeze.json').read_text())
        for name, entry in fr['files'].items():
            p = Path(entry['path'])
            assert sha(p) == entry['sha256'], f'Frozen input changed: {name}'
    d, _ = load_aps_csv(cfg['train_csv']); dt, _ = load_aps_csv(cfg['test_csv'])
    assert list(d.columns) == list(dt.columns)
    x, y = split_features(d); xt, yt = split_features(dt)
    assert x.shape == (60000,170) and xt.shape == (16000,170)
    assert y.sum() == 1000 and yt.sum() == 375
    assert np.isfinite(x[~np.isnan(x)]).all() and np.isfinite(xt[~np.isnan(xt)]).all()
    with np.load(HERE/'split_indices.npz', allow_pickle=False) as z:
        ix = {k:z[k] for k in z.files}
    for s in cfg['split_seeds']:
        tr, va, reserve = [ix[f's{s}_{k}'] for k in ['train','val','reserve']]
        assert np.array_equal(np.sort(np.r_[tr,va,reserve]), np.arange(len(x)))
        assert [len(tr),len(va),len(reserve)] == [42000,9000,9000]
        assert [int(y[a].sum()) for a in [tr,va,reserve]] == [700,150,150]
        with np.load(HERE/f'historical/aps_seed{s}.npz', allow_pickle=False) as old:
            assert np.array_equal(tr, old['train_rows']) and np.array_equal(va, old['val_rows'])
            assert np.array_equal(y[va], old['y_val']) and np.array_equal(yt,old['y_test'])
    features = np.asarray([c for c in d.columns if c != 'class'])
    return x,y,xt,yt,ix,features

def environment():
    return {'python':platform.python_version(), 'executable':sys.executable,
            'platform':platform.platform(), 'numpy':np.__version__, 'pandas':pd.__version__,
            'sklearn':sklearn.__version__, 'scipy':scipy.__version__, 'joblib':joblib.__version__,
            'threadpool_info':threadpool_info()}

def smoke(cfg):
    x,y,xt,yt,ix,features = load(cfg, check_freeze=False)
    dest = HERE/'smoke_results'; dest.mkdir(exist_ok=False)
    tr = ix['s11_train']; tr = np.r_[tr[y[tr]==0][:1960],tr[y[tr]==1][:40]]
    xx,yy = x[tr], y[tr]; med = MedianImputer().fit(xx).medians_
    mask = get_mask(xx, yy, 'C', .25, 9107)
    a3 = get_mask(xx, yy, 'A3', .25, 9107)
    assert np.array_equal(mask.sum(0), a3.sum(0))
    xm=xx.copy(); xm[mask]=np.nan
    m=HistGradientBoostingClassifier(**cfg['model_params']).fit(matrix(xm,med),yy)
    joblib.dump({'model':m,'medians':med},dest/'model.joblib')
    m2=joblib.load(dest/'model.joblib')['model']; ev=build_eval_inputs(xt[:500])
    errs=[float(np.max(np.abs(m.predict_proba(matrix(v[0],med))-m2.predict_proba(matrix(v[0],med))))) for v in ev.values()]
    assert max(errs)==0
    dump(dest/'smoke.json',{'status':'passed','n_train':len(tr),'n_eval':500,'max_reload_error':max(errs),
                           'scope':'interface test only; no AP used for design or historical fidelity',
                           'environment':environment(),'timestamp_utc':stamp()})
    print('SMOKE PASSED',flush=True)

def run(cfg):
    x,y,xt,yt,ix,features = load(cfg)
    out=HERE/'run_results'; out.mkdir(exist_ok=False)
    start=time.perf_counter(); fit_total=0
    dump(out/'environment.json',{**environment(),'started_utc':stamp(),'freeze_sha256':sha(HERE/'freeze.json'),
                                 'effective_model_params':HistGradientBoostingClassifier(**cfg['model_params']).get_params()})
    def log(obj):
        obj={'time_utc':stamp(),**obj}
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        print(json.dumps(obj,ensure_ascii=False),flush=True)
    specs=model_specs(cfg)
    log({'event':'start','expected_fits':len(specs)*len(cfg['split_seeds'])})
    test_evals={k:v[0] for k,v in build_eval_inputs(xt).items()}
    np.savez_compressed(out/'test_evaluation_masks.npz',**{k:np.isnan(v)&~np.isnan(xt) for k,v in test_evals.items()})
    summaries=[]
    try:
        for split in cfg['split_seeds']:
            dest=out/f'split{split}';dest.mkdir()
            for folder in ['models','training_masks','diagnostics','predictions']:(dest/folder).mkdir()
            tr,va=ix[f's{split}_train'],ix[f's{split}_val']
            xx,yy=x[tr],y[tr]; med=MedianImputer().fit(xx).medians_
            q,counts,miss=estimate_class_rates(xx,yy)
            np.savez_compressed(dest/'preprocessing.npz',medians=med,feature_names=features,
                                train_rows=tr,q=q,class_counts=counts,missing_counts=miss)
            evals={k:v[0] for k,v in build_eval_inputs(x[va]).items()}
            np.savez_compressed(dest/'evaluation_masks.npz',**{k:np.isnan(v)&~np.isnan(x[va]) for k,v in evals.items()})
            emats={k:matrix(v,med) for k,v in evals.items()}
            preds={}; fits=[]
            for tag,arm,a,seed in specs:
                mask=get_mask(xx,yy,arm,a,seed)
                assert not (mask & np.isnan(xx)).any()
                xm=xx.copy();xm[mask]=np.nan
                expected=a*(counts[:,None]*(q.max(0)[None,:]-q)).sum(0)
                eb=class_blind_rates(q,counts,a)*(~np.isnan(xx)).sum(0)
                assert np.allclose(expected,eb,rtol=0,atol=cfg['tolerances']['budget_atol'])
                np.savez_compressed(dest/'training_masks'/f'{tag}.npz',new_mask=mask)
                np.savez_compressed(dest/'diagnostics'/f'{tag}.npz',row_deleted=mask.sum(1),col_deleted=mask.sum(0),
                                    class_col_deleted=np.stack([mask[yy==c].sum(0) for c in [0,1]]),
                                    expected_col=expected,reference_col_counts=mask.sum(0),
                                    q_after=estimate_class_rates(xm,yy)[0])
                t=time.perf_counter();model=HistGradientBoostingClassifier(**cfg['model_params'])
                model.fit(matrix(xm,med),yy);fit_seconds=time.perf_counter()-t;fit_total+=1
                modelpath=dest/'models'/f'{tag}.joblib'
                joblib.dump({'model':model,'medians':med,'train_rows':tr,'arm':arm,'alpha':a,'mask_seed':seed,
                             'model_seed':cfg['model_params']['random_state'],'config_sha256':sha(HERE/'config.json')},modelpath,compress=3)
                ps={k:model.predict_proba(v)[:,1] for k,v in emats.items()}
                for p in ps.values():assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                np.savez_compressed(dest/'predictions'/f'{tag}__val.npz',row_indices=va,y=y[va],**ps)
                preds[tag]=ps
                rec={'tag':tag,'n_iter':int(model.n_iter_),'do_early_stopping':bool(model.do_early_stopping_),
                     'fit_seconds':fit_seconds,'model_sha256':sha(modelpath)}
                fits.append(rec);log({'event':'fit','split':split,'fits_completed':fit_total,
                                     'elapsed_seconds':time.perf_counter()-start,**rec})
                del xm,mask,model
            dump(dest/'fit_records.json',fits)
            configs=list(dict.fromkeys(t.split('_seed')[0] for t,_,_,_ in specs))
            ensemble={'train_rows':tr,'val_rows':va,'test_rows':np.arange(len(yt)),'y_val':y[va],'y_test':yt}
            metrics={c:{} for c in configs}
            def aggregate(ps,dataset,labels):
                for c in configs:
                    ts=[t for t,_,_,_ in specs if t.split('_seed')[0]==c]
                    for cond in next(iter(ps.values())):
                        p=np.mean([ps[t][cond] for t in ts],axis=0)
                        ensemble[f'{c}__{dataset}__{cond}']=p
                        metrics[c][f'{dataset}__{cond}']=float(average_precision_score(labels,p))
            aggregate(preds,'val',y[va])
            selection=choose(metrics,cfg);dump(dest/'selection.json',selection)
            log({'event':'selection_frozen','split':split,'chosen':{k:r['alpha'] for k,r in selection['chosen'].items()},
                 'selection_sha256':sha(dest/'selection.json')})
            # No refit: historical APS fit rows are retained. Test is forward evaluation only.
            del emats,evals,preds
            emats={k:matrix(v,med) for k,v in test_evals.items()};preds={}
            for tag,_,_,_ in specs:
                model=joblib.load(dest/'models'/f'{tag}.joblib')['model']
                ps={k:model.predict_proba(v)[:,1] for k,v in emats.items()}
                for p in ps.values():assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                np.savez_compressed(dest/'predictions'/f'{tag}__test.npz',row_indices=np.arange(len(yt)),y=yt,**ps)
                preds[tag]=ps
            aggregate(preds,'test',yt)
            np.savez_compressed(dest/'ensemble_predictions.npz',**ensemble)
            dump(dest/'metrics.json',metrics)
            grouped={}
            for c,values in metrics.items():
                grouped[c]={}
                for ds in ['val','test']:
                    grouped[c][ds]={cond:values[f'{ds}__S0'] if cond=='S0' else
                                    float(np.mean([v for k,v in values.items() if k.startswith(f'{ds}__{cond}_')]))
                                    for cond in ['S0','S1_10','S1_30','S2','S3']}
            for arm in ['C','B']:grouped[arm+'_selected']=grouped[selection['chosen'][arm]['config']]
            summaries.append({'split':split,'chosen_alphas':{k:r['alpha'] for k,r in selection['chosen'].items()},'ap':grouped})
            dump(out/'summary.json',{'status':'running','fits_completed':fit_total,'splits':summaries})
            log({'event':'split_complete','split':split,'fits_completed':fit_total,'elapsed_seconds':time.perf_counter()-start})
            del emats,preds,ensemble
        assert fit_total==cfg['expected_models']
        means={c:{ds:{cond:float(np.mean([s['ap'][c][ds][cond] for s in summaries]))
                     for cond in ['S0','S1_10','S1_30','S2','S3']} for ds in ['val','test']}
               for c in summaries[0]['ap']}
        dump(out/'summary.json',{'status':'complete','fits_completed':fit_total,'splits':summaries,
                                'equal_split_mean_ap':means,'finished_utc':stamp(),'wall_seconds':time.perf_counter()-start,
                                'inference':'descriptive, overlapping splits; no new confidence intervals'})
        log({'event':'complete','fits_completed':fit_total,'wall_seconds':time.perf_counter()-start})
    except BaseException as e:
        dump(out/'failure.json',{'timestamp_utc':stamp(),'fits_completed':fit_total,'error':repr(e),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=json.loads((HERE/'config.json').read_text())
    with threadpool_limits(limits=2):
        if args.check:
            x,y,xt,yt,ix,features=load(cfg)
            print(json.dumps({'status':'passed','train_shape':x.shape,'test_shape':xt.shape,'expected_fits':125,'environment':environment()},default=str))
        elif args.smoke:smoke(cfg)
        else:run(cfg)
