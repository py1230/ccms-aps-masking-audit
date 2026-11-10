"""Four-fit environment diagnostic, isolated from the main APS rerun."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys
sys.dont_write_bytecode=True
import argparse,csv,hashlib,json,platform,subprocess,time,traceback
from importlib import metadata
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd,scipy,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as AP
from threadpoolctl import threadpool_limits,threadpool_info

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path[:0]=[str(BASE/'source'),str(BASE/'historical')]
from aps_io import load_aps_csv,split_features
from pipeline import MedianImputer
from ccms import mask_class_conditional
from run_cb_firstlook import build_eval_inputs

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def matrix(x,med):
    return np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])
def environment():
    return dict(python=platform.python_version(),executable=sys.executable,
                numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
                scipy=scipy.__version__,joblib=joblib.__version__,platform=platform.platform(),
                threads=2,threadpool_info=threadpool_info())
def prepare():
    assert not (HERE/'freeze.json').exists(), 'Refuse to replace existing freeze'
    parent=json.loads((BASE/'config.json').read_text())
    cfg={'split_seed':11,'mask_seeds':[9107,9108,9109],'alpha':.25,
         'model_params':parent['model_params'],'train_csv':parent['train_csv'],'test_csv':parent['test_csv'],
         'prediction_atol':1e-12,'ap_atol':1e-12,'n_fits':4,
         'expected_environment':{'python':'3.12.15','numpy':'2.5.4','sklearn':'1.9.1'},
         'scope':'post-outcome bounded diagnostic; no parameter adaptation'}
    dump(HERE/'config.json',cfg)
    defaults=HistGradientBoostingClassifier(random_state=2026).get_params()
    diffs={k:{'explicit_main_config':cfg['model_params'].get(k),'historical_environment_default':defaults.get(k)}
           for k in set(defaults)|set(cfg['model_params']) if defaults.get(k)!=cfg['model_params'].get(k)}
    dump(HERE/'environment_and_defaults.json',{'environment':environment(),'defaults':defaults,
         'effective_explicit_params':HistGradientBoostingClassifier(**cfg['model_params']).get_params(),
         'parameter_differences':diffs,'timestamp_utc':stamp()})
    proc=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=False)
    if proc.returncode==0:
        (HERE/'environment_packages.txt').write_text(proc.stdout)
    else:
        (HERE/'pip_freeze_unavailable.txt').write_text(proc.stderr)
        packages=sorted({f'{d.metadata["Name"]}=={d.version}' for d in metadata.distributions() if d.metadata['Name']})
        (HERE/'environment_packages.txt').write_text('# Snapshot from importlib.metadata; pip unavailable\n'+'\n'.join(packages)+'\n')
    files=[HERE/'PROTOCOL.md',HERE/'run_probe.py',HERE/'config.json',HERE/'environment_and_defaults.json',
           HERE/'environment_packages.txt',BASE/'config.json',BASE/'freeze.json',BASE/'split_indices.npz',
           BASE/'historical/aps_seed11.npz',BASE/'run_results/split11/ensemble_predictions.npz',
           BASE/'run_results/split11/preprocessing.npz',Path(cfg['train_csv']),Path(cfg['test_csv']),Path(sys.executable)]
    files+=list((BASE/'source').glob('*.py'))
    files+=[BASE/'historical/run_cb_firstlook.py']
    files+=[BASE/f'run_results/split11/training_masks/C025_seed{m}.npz' for m in cfg['mask_seeds']]
    dump(HERE/'freeze.json',{'frozen_utc':stamp(),'planned_fits':4,
         'prior_outcome_access':'Historical outputs and main-run differences seen; no probe models fitted yet',
         'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]})
    print('PREPARED',json.dumps({'parameter_differences':diffs,'n_frozen_files':len(files)}),flush=True)
def run():
    fr=json.loads((HERE/'freeze.json').read_text())
    for x in fr['files']:assert sha(x['path'])==x['sha256'],x['path']
    cfg=json.loads((HERE/'config.json').read_text());env=environment()
    for k,v in cfg['expected_environment'].items():assert env[k]==v,(k,env[k],v)
    out=HERE/'results';out.mkdir(exist_ok=False)
    for folder in ['models','masks','predictions']:(out/folder).mkdir()
    dump(out/'environment.json',env)
    start=time.perf_counter();nfit=0
    def log(o):
        rec={'timestamp_utc':stamp(),**o}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)
    try:
        df,_=load_aps_csv(cfg['train_csv']);dt,_=load_aps_csv(cfg['test_csv'])
        assert list(df.columns)==list(dt.columns)
        X,y=split_features(df);xt,yt=split_features(dt)
        assert X.shape==(60000,170) and xt.shape==(16000,170)
        ix=np.load(BASE/'split_indices.npz');tr=ix['s11_train'];va=ix['s11_val'];reserve=ix['s11_reserve']
        assert np.array_equal(np.sort(np.r_[tr,va,reserve]),np.arange(60000))
        old=np.load(BASE/'historical/aps_seed11.npz');new=np.load(BASE/'run_results/split11/ensemble_predictions.npz')
        assert np.array_equal(tr,old['train_rows']) and np.array_equal(va,old['val_rows'])
        xx,yy=X[tr],y[tr];med=MedianImputer().fit(xx).medians_
        mainprep=np.load(BASE/'run_results/split11/preprocessing.npz')
        assert np.array_equal(med,mainprep['medians'])
        evals={f'val__{k}':v[0] for k,v in build_eval_inputs(X[va]).items()}
        evals.update({f'test__{k}':v[0] for k,v in build_eval_inputs(xt).items()})
        np.savez_compressed(out/'preprocessing.npz',medians=med,train_rows=tr,val_rows=va,
                            test_rows=np.arange(len(yt)),feature_names=np.asarray([c for c in df.columns if c!='class']))
        np.savez_compressed(out/'evaluation_masks.npz',**{k:np.isnan(v)&~np.isnan(X[va] if k.startswith('val__') else xt) for k,v in evals.items()})
        emats={k:matrix(v,med) for k,v in evals.items()};preds={};records=[];mask_checks=[]
        specs=[('M3',None)]+[(f'C025_seed{ms}',ms) for ms in cfg['mask_seeds']]
        log({'event':'start','n_fits':4,'freeze_sha256':sha(HERE/'freeze.json')})
        for tag,seed in specs:
            if seed is None:xm=xx.copy();mask=np.zeros(xx.shape,dtype=bool)
            else:
                xm,info=mask_class_conditional(xx,yy,.25,seed=seed);mask=info['new_mask']
                mainmask=np.load(BASE/f'run_results/split11/training_masks/{tag}.npz')['new_mask']
                assert np.array_equal(mask,mainmask)
                mask_checks.append({'tag':tag,'cellwise_equal_to_main':True,'deleted':int(mask.sum())})
            assert not (mask&np.isnan(xx)).any()
            np.savez_compressed(out/'masks'/f'{tag}.npz',new_mask=mask,row_deleted=mask.sum(1),col_deleted=mask.sum(0))
            model=HistGradientBoostingClassifier(**cfg['model_params']);t=time.perf_counter();model.fit(matrix(xm,med),yy)
            fit_sec=time.perf_counter()-t;nfit+=1
            modelpath=out/'models'/f'{tag}.joblib';joblib.dump({'model':model,'medians':med,'mask_seed':seed,'train_rows':tr},modelpath,compress=3)
            ps={k:model.predict_proba(v)[:,1] for k,v in emats.items()}
            for p in ps.values():assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            np.savez_compressed(out/'predictions'/f'{tag}.npz',train_rows=tr,val_rows=va,test_rows=np.arange(len(yt)),y_val=y[va],y_test=yt,**ps)
            reloaded=joblib.load(modelpath)['model']
            re_err=max(float(np.max(np.abs(reloaded.predict_proba(v)[:,1]-ps[k]))) for k,v in emats.items())
            assert re_err<=cfg['prediction_atol']
            preds[tag]=ps
            rec={'tag':tag,'fit_seconds':fit_sec,'n_iter':int(model.n_iter_),'do_early_stopping':bool(model.do_early_stopping_),
                 'max_reload_prediction_error':re_err,'model_sha256':sha(modelpath)}
            records.append(rec);log({'event':'fit_complete','fits':nfit,**rec})
        ensemble={'M3':preds['M3'],'C025':{k:np.mean([preds[f'C025_seed{ms}'][k] for ms in cfg['mask_seeds']],axis=0) for k in emats}}
        np.savez_compressed(out/'ensemble_predictions.npz',train_rows=tr,val_rows=va,y_val=y[va],y_test=yt,
                            **{f'{c}__{k}':p for c,z in ensemble.items() for k,p in z.items()})
        rows=[]
        for c,z in ensemble.items():
            for k,p in z.items():
                ds,condition=k.split('__');labels=y[va] if ds=='val' else yt;key=f'{c}__{k}'
                assert np.array_equal(labels,old[f'y_{ds}']) and np.array_equal(labels,new[f'y_{ds}'])
                ap=float(AP(labels,p));op=old[key];npred=new[key]
                rows.append({'config':c,'dataset':ds,'condition':condition,'probe_ap':ap,
                             'historical_ap':float(AP(labels,op)),'main_new_ap':float(AP(labels,npred)),
                             'max_abs_prediction_difference_historical':float(np.max(np.abs(p-op))),
                             'max_abs_prediction_difference_main':float(np.max(np.abs(p-npred))),
                             'ap_difference_historical':ap-float(AP(labels,op)),
                             'ap_difference_main':ap-float(AP(labels,npred))})
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        byconfig={}
        for c in ensemble:
            rr=[r for r in rows if r['config']==c]
            byconfig[c]={'n_prediction_arrays':len(rr),'n_match_historical_at_1e12':sum(r['max_abs_prediction_difference_historical']<=1e-12 for r in rr),
                         'max_abs_prediction_difference_historical':max(r['max_abs_prediction_difference_historical'] for r in rr),
                         'max_abs_prediction_difference_main':max(r['max_abs_prediction_difference_main'] for r in rr),
                         'max_abs_ap_difference_historical':max(abs(r['ap_difference_historical']) for r in rr),
                         'max_abs_ap_difference_main':max(abs(r['ap_difference_main']) for r in rr)}
        for x in fr['files']:assert sha(x['path'])==x['sha256'],x['path']
        summary={'status':'complete','n_fits':nfit,'n_comparison_arrays':len(rows),'by_config':byconfig,
                 'fit_records':records,'masks_equal_main':mask_checks,'medians_equal_main':True,
                 'wall_seconds':time.perf_counter()-start,'finished_utc':stamp(),'environment':env,
                 'scope':'split11 M3 and C025 only; joint software environment comparison, no individual-library attribution or historical training provenance claim'}
        dump(out/'summary.json',summary);log({'event':'complete','by_config':byconfig})
        paths=[p for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json' and p.suffix not in ['.log']]
        dump(HERE/'manifest.json',{'files':[{'path':str(p.relative_to(HERE)),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]})
    except BaseException as e:
        dump(out/'failure.json',{'error':repr(e),'fits_completed':nfit,'traceback':traceback.format_exc(),'timestamp_utc':stamp()})
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=2):prepare() if args.prepare else run()
