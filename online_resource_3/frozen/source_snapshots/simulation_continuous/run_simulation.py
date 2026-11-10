"""New, fully archived continuation of the historical continuous-missingness simulation.

Source generator/operator bytes are retained in source/. New wrapper separates source
selection from target prediction and refuses existing output directories.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse, hashlib, json, platform, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import scipy, sklearn, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as AP
from threadpoolctl import threadpool_limits, threadpool_info
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from sim_nuisance import gen_data, base_rates
from ccms import estimate_class_rates, class_blind_rates, mask_class_conditional, mask_class_blind
from pipeline import MedianImputer

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def stamp(): return datetime.now(timezone.utc).isoformat()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
            'scipy':scipy.__version__,'joblib':joblib.__version__,'executable':sys.executable,
            'platform':platform.platform(),'threadpools':threadpool_info()}

def check(cfg,frozen=True):
    for k,v in cfg['input_environment'].items(): assert environment()[k]==v,(k,environment()[k],v)
    assert cfg['model_params']==HistGradientBoostingClassifier(random_state=cfg['model_seed']).get_params()
    assert cfg['data_seeds']==list(range(601,631)) and cfg['smoke_seed'] not in cfg['data_seeds']
    assert cfg['expected_models']==len(cfg['data_seeds'])*cfg['expected_models_per_seed']
    if frozen:
        fr=json.loads((HERE/'freeze.json').read_text())
        for name,h in fr['files'].items(): assert sha(HERE/name)==h, name

def matrix(x,med,tag):
    return x if tag=='M4' else np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])

def split(y,seed):
    rng=np.random.default_rng(seed+7);parts=[[],[],[]]
    for c in [0,1]:
        idx=rng.permutation(np.flatnonzero(y==c));i,j=round(.6*len(idx)),round(.8*len(idx))
        for k,z in enumerate([idx[:i],idx[i:j],idx[j:]]):parts[k].extend(z)
    return [np.sort(v) for v in parts]

def natural(xx,yy,b,delta,m,U):
    pi=yy.mean();q1=b+(1-pi)*delta*m;q0=b-pi*delta*m
    assert min(q1.min(),q0.min())>=0 and max(q1.max(),q0.max())<=1
    mask=U<np.where(yy[:,None]==1,q1,q0);z=xx.copy();z[mask]=np.nan
    return z,mask,{'q1':q1.tolist(),'q0':q0.tolist(),'prevalence':float(pi),
                  'expected_overall_missing':float((pi*q1+(1-pi)*q0).mean()),
                  'actual_missing':float(mask.mean())}

def specifications(cfg):
    specs=[('M3','M3',0.,None),('M4','M4',0.,None)]
    for a in cfg['alphas'][1:]:
        for arm in ['C','B']:
            for ms in cfg['mask_seeds']:specs.append((f'{arm}{round(a*100):03d}_seed{ms}',arm,a,ms))
    return specs

def ensembles(single,cfg):
    result={k:single[k] for k in ['M3','M4']}
    for a in cfg['alphas'][1:]:
        for arm in ['C','B']:
            tag=f'{arm}{round(a*100):03d}'
            result[tag]={k:np.mean([single[f'{tag}_seed{ms}'][k] for ms in cfg['mask_seeds']],axis=0)
                         for k in single['M3']}
    return result

def select(y,ps,cfg):
    base=float(AP(y,ps['M3']['S0']));out={}
    for arm in ['C','B']:
        cand=[]
        for a in cfg['alphas']:
            tag='M3' if a==0 else f'{arm}{round(a*100):03d}'
            vals=[float(AP(y,ps[tag][k])) for k in cfg['selection_conditions']]
            cand.append({'alpha':a,'config':tag,'s0':vals[0],'worst4':min(vals),
                         'eligible':bool(vals[0]>=base-cfg['selection_s0_tolerance'])})
        out[arm]={'chosen':max([v for v in cand if v['eligible']],key=lambda v:(v['worst4'],-v['alpha'])),
                   'candidates':cand}
    return out

def run_seed(ds,cfg,smoke=False):
    check(cfg,frozen=not smoke)
    root=HERE/('smoke_results' if smoke else 'run_results')
    root.mkdir(exist_ok=True);out=root/f'seed{ds}';out.mkdir(exist_ok=False)
    for name in ['models','masks','predictions']: (out/name).mkdir()
    started=time.time();fits=[]
    def event(kind,**kw):
        row={'timestamp_utc':stamp(),'event':kind,'data_seed':ds,**kw}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row),flush=True)
    try:
        event('started',pid=os.getpid(),smoke=smoke,config_sha256=sha(HERE/'config.json'))
        dump(out/'environment.json',environment())
        X,y,meta=gen_data(ds);assert X.shape==(cfg['n'],cfg['p'])
        tr,va,te=split(y,ds)
        assert np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(y)))
        b,delta=base_rates(ds);b[:6]=.4
        rng=np.random.default_rng(ds);assert np.array_equal(rng.normal(size=X.shape),X)
        ul=rng.random(len(y));s=1.2*(X[:,0]+X[:,1]+X[:,2])
        assert np.array_equal(y,(ul<1/(1+np.exp(-(s-meta['intercept'])))).astype(int))
        U={f'U_{name}':np.random.default_rng(ds*10+k).random((len(rows),X.shape[1]))
           for name,k,rows in [('train',1,tr),('val',2,va),('test',3,te)]}
        U.update({f'U_mask_{ms}':np.random.default_rng(ms).random((len(tr),X.shape[1])) for ms in cfg['mask_seeds']})
        U.update({f'U_stress_{ss}':np.random.default_rng(ss).random((len(va),X.shape[1])) for ss in cfg['stress_seeds']})
        np.savez_compressed(out/'data.npz',X=X,y=y,train_rows=tr,val_rows=va,test_rows=te,b=b,delta=delta,U_label=ul,**U)
        x,mt,mtrain=natural(X[tr],y[tr],b,delta,1,U['U_train'])
        v,mv,mval=natural(X[va],y[va],b,delta,1,U['U_val'])
        ve={'S0':v};masks={'train_natural':mt,'val_natural':mv}
        for ss in cfg['stress_seeds']:
            new=(~np.isnan(v)) & (U[f'U_stress_{ss}']<.1)
            vv=v.copy();vv[new]=np.nan;ve[f'S1_10_{ss}']=vv;masks[f'val_S1_10_{ss}']=new
        med=MedianImputer().fit(x).medians_;q,cnt,mc=estimate_class_rates(x,y[tr])
        np.savez_compressed(out/'preprocessing.npz',medians=med,q=q,counts=cnt,missing_counts=mc)
        models={};single_val={}
        for tag,arm,a,ms in specifications(cfg):
            t=time.time();event('fit_started',tag=tag)
            if ms is None:
                xm=x.copy();new=np.zeros(x.shape,dtype=bool);expected=np.zeros(x.shape[1]);bv=np.zeros(x.shape[1])
            else:
                bv=class_blind_rates(q,cnt,a)
                if arm=='C':xm,info=mask_class_conditional(x,y[tr],a,uniforms=U[f'U_mask_{ms}'])
                else:xm,info=mask_class_blind(x,bv,uniforms=U[f'U_mask_{ms}'])
                new=info['new_mask'];expected=a*(cnt[:,None]*(q.max(0)-q)).sum(0)
                assert np.allclose(expected,bv*(~np.isnan(x)).sum(0),atol=1e-8,rtol=0)
            assert not np.any(new & np.isnan(x))
            np.savez_compressed(out/'masks'/f'{tag}.npz',new_mask=new,expected_per_col=expected,
                                blind_probabilities=bv,row_deleted=new.sum(1),col_deleted=new.sum(0),
                                class_col_deleted=np.array([new[y[tr]==c].sum(0) for c in [0,1]]))
            model=HistGradientBoostingClassifier(**cfg['model_params']).fit(matrix(xm,med,tag),y[tr])
            mp=out/'models'/f'{tag}.joblib'
            joblib.dump({'model':model,'medians':med,'tag':tag,'arm':arm,'alpha':a,'mask_seed':ms,
                         'data_seed':ds,'train_rows':tr,'model_seed':cfg['model_seed'],
                         'config_sha256':sha(HERE/'config.json')},mp,compress=3)
            models[tag]=model
            single_val[tag]={k:model.predict_proba(matrix(xx,med,tag))[:,1] for k,xx in ve.items()}
            np.savez_compressed(out/'predictions'/f'{tag}__val.npz',row_indices=va,y=y[va],**single_val[tag])
            fits.append({'tag':tag,'model_sha256':sha(mp),'params':model.get_params(),
                         'n_iter':int(model.n_iter_),'seconds':time.time()-t,'status':'success'})
            dump(out/'fit_records.json',fits);event('fit_completed',tag=tag,seconds=time.time()-t)
        vp=ensembles(single_val,cfg);selection=select(y[va],vp,cfg)
        dump(out/'selection.json',selection)
        event('selection_saved',selection_sha256=sha(out/'selection.json'),
              chosen={a:selection[a]['chosen']['alpha'] for a in ['C','B']})
        # No target construction or prediction occurs until source selection is durable.
        ee={};target_meta={}
        for m in cfg['target_gap_multipliers']:
            k=f'gap_{m:g}';ee[k],masks['target_'+k],target_meta[k]=natural(X[te],y[te],b,delta,m,U['U_test'])
        np.savez_compressed(out/'evaluation_masks.npz',**masks)
        event('target_prediction_started')
        single_test={}
        for tag,model in models.items():
            single_test[tag]={k:model.predict_proba(matrix(xx,med,tag))[:,1] for k,xx in ee.items()}
            np.savez_compressed(out/'predictions'/f'{tag}__test.npz',row_indices=te,y=y[te],**single_test[tag])
        tp=ensembles(single_test,cfg)
        for arm in ['C','B']:
            chosen=selection[arm]['chosen']['config'];vp[arm+'_selected']=vp[chosen];tp[arm+'_selected']=tp[chosen]
        payload={'y_val':y[va],'y_test':y[te],'val_rows':va,'test_rows':te,'train_rows':tr}
        metrics={}
        for tag in vp:
            metrics[tag]={}
            for part,probs,yy in [('val',vp[tag],y[va]),('test',tp[tag],y[te])]:
                metrics[tag][part]={k:float(AP(yy,p)) for k,p in probs.items()}
                for k,p in probs.items():payload[f'{tag}__{part}__{k}']=p
        np.savez_compressed(out/'ensemble_predictions.npz',**payload);dump(out/'metrics.json',metrics)
        if smoke:
            errors={}
            for tag in models:
                reloaded=joblib.load(out/'models'/f'{tag}.joblib')['model']
                errors[tag]=max(float(np.max(np.abs(reloaded.predict_proba(matrix(xx,med,tag))[:,1]-single_test[tag][k]))) for k,xx in ee.items())
            assert max(errors.values())==0
            tested=['run_simulation.py','launch_simulation.py','freeze_run.py','config.json']
            tested += [str(p.relative_to(HERE)) for folder in ['source','historical'] for p in sorted((HERE/folder).glob('*.py'))]
            dump(out/'smoke_reload_check.json',{'status':'passed','models':len(errors),'maximum_error':max(errors.values()),
                                               'data_seed':ds,'tested_files':{name:sha(HERE/name) for name in tested}})
        result={'status':'complete','data_seed':ds,'smoke':smoke,'data_meta':meta,'source_selection':selection,
                'train_missing':mtrain,'validation_missing':mval,'target_missing':target_meta,
                'metrics':{k:z['test'] for k,z in metrics.items()},'fits_completed':len(fits),
                'seconds':time.time()-started,'finished_at_utc':stamp()}
        assert len(fits)==cfg['expected_models_per_seed']
        event('complete',fits_completed=len(fits),seconds=time.time()-started)
        manifest={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size}
                  for p in sorted(out.rglob('*')) if p.is_file()}
        dump(out/'manifest.json',manifest);result['manifest_sha256']=sha(out/'manifest.json')
        dump(out/'result.json',result)
        return result
    except BaseException as e:
        event('failed',error=repr(e),fits_completed=len(fits))
        dump(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'fits_completed':len(fits),
                                 'timestamp_utc':stamp(),'automatic_retry':False})
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--check',action='store_true')
    args=ap.parse_args();cfg=json.loads((HERE/'config.json').read_text())
    with threadpool_limits(cfg['threads']):
        if args.check:check(cfg);print('Frozen files and runtime verified');return
        if args.smoke:
            assert args.seed is None;run_seed(cfg['smoke_seed'],cfg,True)
        else:
            assert args.seed in cfg['data_seeds'];run_seed(args.seed,cfg)
if __name__=='__main__':main()
