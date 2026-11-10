"""Finite-attempt A5 randomization and paired, descriptive APS experiment."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import argparse, ctypes, hashlib, json, platform, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from aps_io import load_aps_csv,split_features
GROUPS=['S0','S1_10','S1_30','S2','S3']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def cfg():return json.loads((HERE/'config.json').read_text())
def env():return {'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'executable':sys.executable,'platform':platform.platform()}
def compile_lib():
    target=HERE/'source/switch.dylib'
    cmd=['cc','-O3','-shared','-fPIC',str(HERE/'source/switch.c'),'-o',str(target)]
    subprocess.run(cmd,check=True)
    dump(HERE/'compiler.json',{'command':cmd,'version':subprocess.check_output(['cc','--version'],text=True),'library_sha256':sha(target),'source_sha256':sha(HERE/'source/switch.c'),'timestamp':stamp()})
def library():
    lib=ctypes.CDLL(str(HERE/'source/switch.dylib'))
    ptr=ctypes.c_void_p
    lib.a5_switch.argtypes=[ptr,ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_uint32,ptr]
    lib.a5_switch.restype=ctypes.c_size_t
    return lib
def data():
    d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');x,y=split_features(d)
    dt,_=load_aps_csv(HERE/'raw_data/aps_failure_test_set.csv');xt,yt=split_features(dt)
    return x,y,xt,yt
def prepare():
    assert not (HERE/'config.json').exists()
    origin=HERE.parent/'aps_channels_repeats_20261108'
    old=json.loads((origin/'config.json').read_text())
    config={'design':'finite-step double-margin matched A5; descriptive conditional randomization',
      'split_seeds':old['split_seeds'],'reference_mask_seeds':[19107,19108,19109],
      'randomization_repeats':[1,2,3,4,5],'model_params':old['model_params'],
      'input_environment':old['input_environment'],'channel':'M3','reference_repeat':1,'alpha':.25,
      'attempts_per_edge':20,'checkpoints_per_edge':[0,1,2,5,10,20],'batch_size':200000,
      'randomization_seed_formula':'810000000 + split*100000 + reference_seed*10 + repeat',
      'primary_contrast':'C025 minus A5, test S1_30; mean pressure AP then mean 5 splits',
      'other_comparisons':'C025-A5, A5-B025, A5-BASE in five groups; descriptive only',
      'statistics':'mean and min/max over five independent randomization repetitions; no CI or p values',
      'expected_fits':75,'threads':2,'max_run_seconds':7200,
      'feasibility_gate':{'minimum_reference_edge_relocated_fraction':.5,'all_pairwise_hamming_positive':True,
                          'minimum_disjoint_switch_certificates':100},
      'dtype_array_hash':'SHA256(dtype.str UTF8 + compact JSON list(shape) UTF8 + C order bytes)',
      'rng_calls_per_batch':'rng.integers(0,E,n,dtype=uint32) for a; same for b; rng.integers(0,2,n,dtype=uint8) for lazy',
      'source':'APS channels M3 rep1 matched environment; old performance previously available; no results used for A5 design'}
    dump(HERE/'config.json',config)
    (HERE/'raw_data').mkdir()
    for name in ['train_csv','test_csv']:shutil.copy2(old[name],HERE/'raw_data'/Path(old[name]).name)
    root=origin/'run_results/M3'
    (HERE/'reference').mkdir()
    shutil.copy2(root/'test_evaluation_masks.npz',HERE/'reference/test_evaluation_masks.npz')
    sources={}
    for split in config['split_seeds']:
        src=root/'rep1'/f'split{split}';dest=HERE/'reference'/f'split{split}';dest.mkdir()
        for sub in ['training_masks','predictions','models']:(dest/sub).mkdir()
        for name in ['preprocessing.npz','evaluation_masks.npz']:
            shutil.copy2(src/name,dest/name);sources[str((dest/name).relative_to(HERE))]={'source':str(src/name),'sha256':sha(src/name)}
        for tag in ['BASE']+[f'{a}025_seed{s}' for a in ['C','B'] for s in config['reference_mask_seeds']]:
            for sub,suffix in [('training_masks','.npz'),('models','.joblib'),('predictions','__test.npz')]:
                shutil.copy2(src/sub/(tag+suffix),dest/sub/(tag+suffix))
                sources[str((dest/sub/(tag+suffix)).relative_to(HERE))]={'source':str(src/sub/(tag+suffix)),'sha256':sha(src/sub/(tag+suffix))}
    dump(HERE/'reference_sources.json',sources);compile_lib()
    dump(HERE/'environment.json',env())
    print('PREPARED',flush=True)
def diagnostics(mask,ref,allowed,attempt,accepted):
    edge_count=int(ref.sum());common=int((mask&ref).sum())
    cols=np.flatnonzero(ref.sum(0))[:16]
    small=mask[:,cols].astype(np.float64);co=small.T@small
    return {'attempt':int(attempt),'accepted':int(accepted),'acceptance_rate':accepted/attempt if attempt else 0.,
      'common_reference_edges':common,'relocated_fraction':(edge_count-common)/edge_count,
      'hamming':int(np.count_nonzero(mask!=ref)),'forbidden_count':int((mask&~allowed).sum()),
      'row_margin_max_error':int(np.abs(mask.sum(1).astype(int)-ref.sum(1)).max()),
      'col_margin_max_error':int(np.abs(mask.sum(0).astype(int)-ref.sum(0)).max()),
      'diagnostic_cols':cols.tolist(),'cooccurrence_upper':co[np.triu_indices(len(cols),1)].astype(int).tolist(),
      'mask_sha256':arrsha(mask.astype(bool))}
def certificate(ref,allowed,seed):
    rng=np.random.default_rng(seed);rows,cols=np.nonzero(ref);used=set();rect=[]
    for _ in range(200000):
        i,j=rng.integers(len(rows),size=2);r,s=int(rows[i]),int(rows[j]);c,d=int(cols[i]),int(cols[j])
        if r==s or c==d or r in used or s in used:continue
        if ref[r,d] or ref[s,c] or not allowed[r,d] or not allowed[s,c]:continue
        rect.append([r,s,c,d]);used.update([r,s])
        if len(rect)==256:break
    return np.array(rect,dtype=np.uint32)
def chain(ref,allowed,seed,out,config):
    out.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter();rng=np.random.default_rng(seed);initial=rng.bit_generator.state
    mask=np.ascontiguousarray(ref,dtype=np.uint8);allowed=np.ascontiguousarray(allowed,dtype=np.uint8)
    rr,cc=np.nonzero(mask);rows=rr.astype(np.uint32);cols=cc.astype(np.uint32);e=len(rows)
    lib=library();accepted=0;attempt=0;traces=[];batches=[]
    diag=[diagnostics(mask.astype(bool),ref,allowed.astype(bool),0,0)]
    for point in config['checkpoints_per_edge'][1:]:
        stop=int(point*e)
        while attempt<stop:
            n=min(config['batch_size'],stop-attempt);before=rng.bit_generator.state
            a=rng.integers(0,e,n,dtype=np.uint32);b=rng.integers(0,e,n,dtype=np.uint32);lazy=rng.integers(0,2,n,dtype=np.uint8)
            trace=np.empty((n,3),dtype=np.uint32)
            used=lib.a5_switch(mask.ctypes.data,allowed.ctypes.data,rows.ctypes.data,cols.ctypes.data,a.ctypes.data,b.ctypes.data,lazy.ctypes.data,n,mask.shape[1],attempt,trace.ctypes.data)
            traces.append(trace[:used].copy());accepted+=used
            batches.append({'start':attempt,'n':n,'rng_before':before,'rng_after':rng.bit_generator.state,'a_sha256':arrsha(a),'b_sha256':arrsha(b),'lazy_sha256':arrsha(lazy),'accepted':used})
            attempt+=n
        diag.append(diagnostics(mask.astype(bool),ref,allowed.astype(bool),attempt,accepted))
    final=mask.astype(bool)
    assert attempt==config['attempts_per_edge']*e
    assert not (final&~allowed.astype(bool)).any()
    assert np.array_equal(final.sum(0),ref.sum(0)) and np.array_equal(final.sum(1),ref.sum(1))
    np.savez_compressed(out/'mask.npz',new_mask=final,row_deleted=final.sum(1),col_deleted=final.sum(0))
    np.savez_compressed(out/'trace.npz',accepted_trace=np.concatenate(traces),initial_edge_rows=rows,initial_edge_cols=cc.astype(np.uint32))
    dump(out/'chain.json',{'seed':seed,'edge_count':e,'attempts':attempt,'accepted':accepted,'rng_initial':initial,'rng_final':rng.bit_generator.state,'batches':batches,'diagnostics':diag,'seconds':time.perf_counter()-t,'mask_sha256':arrsha(final),'config_sha256':sha(HERE/'config.json')})
    return final,diag
def feasibility(pilot=False):
    config=cfg();d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');x,y=split_features(d)
    root=HERE/('pilot' if pilot else 'feasibility');root.mkdir(exist_ok=False)
    splits=config['split_seeds'][:1] if pilot else config['split_seeds']
    seeds=config['reference_mask_seeds'][:1] if pilot else config['reference_mask_seeds']
    reps=[0] if pilot else config['randomization_repeats'];results=[]
    for split in splits:
        refroot=HERE/'reference'/f'split{split}';pr=np.load(refroot/'preprocessing.npz');allowed=~np.isnan(x[pr['train_rows']])
        for rs in seeds:
            ref=np.load(refroot/'training_masks'/f'C025_seed{rs}.npz')['new_mask'];dest=root/f'split{split}'/f'reference{rs}';dest.mkdir(parents=True)
            cert=certificate(ref,allowed,700000+split+rs);np.savez_compressed(dest/'legal_switch_certificate.npz',rectangles=cert)
            masks=[]
            for rep in reps:
                seed=810000000+split*100000+rs*10+rep
                m,d=chain(ref,allowed,seed,dest/f'rep{rep}',config);masks.append(m)
                row={'split':split,'reference_seed':rs,'repeat':rep,'seed':seed,'edge_count':int(ref.sum()),'certificate_count':len(cert),'final':d[-1]};results.append(row)
                dump(root/'progress.json',results);print(json.dumps(row),flush=True)
            pairwise=[int(np.count_nonzero(masks[i]!=masks[j])) for i in range(len(masks)) for j in range(i)]
            dump(dest/'diversity.json',{'pairwise_hamming':pairwise,'distinct_masks':len({arrsha(m) for m in masks}),'legal_space_lower_bound':f'2^{len(cert)}','certificate':'row-disjoint legal rectangles; all subsets preserve margins and forbidden cells'})
    passed=all(r['final']['relocated_fraction']>=config['feasibility_gate']['minimum_reference_edge_relocated_fraction'] and r['certificate_count']>=config['feasibility_gate']['minimum_disjoint_switch_certificates'] for r in results)
    if not pilot:
        for p in root.glob('split*/reference*/diversity.json'):passed=passed and all(v>0 for v in json.loads(p.read_text())['pairwise_hamming'])
    dump(root/'summary.json',{'status':'passed' if passed else 'insufficient','performance_accessed':False,'chains':len(results),'rows':results,'timestamp':stamp()})
def freeze():
    assert json.loads((HERE/'feasibility/summary.json').read_text())['status']=='passed'
    assert not (HERE/'freeze.json').exists()
    files={}
    allow_files={'run_a5.py','summarize_results.py','config.json','PROTOCOL.md','compiler.json','environment.json','reference_sources.json','source_config.json','split_indices.npz','pilot_decision.json'}
    allow_roots={'source','raw_data','reference','feasibility'}
    for name in allow_files:assert (HERE/name).is_file(),name
    for name in allow_roots:assert (HERE/name).is_dir(),name
    assert len(list((HERE/'feasibility').glob('split*/reference*/rep*/mask.npz')))==75
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.log':continue
        if str(rel) not in allow_files and rel.parts[0] not in allow_roots:continue
        files[str(rel)]=sha(p)
    dump(HERE/'freeze.json',{'timestamp':stamp(),'environment':env(),'files':files,'scope':'all inputs, references, source, config, protocol and 75 prespecified masks frozen before any A5 performance'})
    print('FROZEN',flush=True)
def check():
    for name,h in json.loads((HERE/'freeze.json').read_text())['files'].items():assert sha(HERE/name)==h,name
    for k,v in cfg()['input_environment'].items():assert env()[k]==v,(k,env()[k],v)
def matrix(x,med):return np.hstack([np.where(np.isnan(x),med[None,:],x),np.isnan(x).astype(float)])
def grouped(vals):return {g:float(vals['S0'] if g=='S0' else np.mean([v for k,v in vals.items() if k.startswith(g+'_')])) for g in GROUPS}
def run(split):
    check();config=cfg();assert split in config['split_seeds'];x,y,xt,yt=data()
    root=HERE/'run_results'/f'split{split}';root.mkdir(parents=True,exist_ok=False)
    def log(o):
        o={'timestamp':stamp(),**o}
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(o)+'\n')
        print(json.dumps(o),flush=True)
    start=time.perf_counter();fits=0
    try:
        refroot=HERE/'reference'/f'split{split}';pr=np.load(refroot/'preprocessing.npz');tr=pr['train_rows'];med=pr['medians'];xx=x[tr];yy=y[tr]
        # Verify preprocessing rather than silently refit or change it.
        with np.errstate(all='ignore'):calc=np.nanmedian(xx,axis=0)
        calc=np.nan_to_num(calc,nan=0.);assert np.array_equal(calc,med)
        ev=np.load(HERE/'reference/test_evaluation_masks.npz');conditions=list(ev.files)
        reference_predictions={};reference_metrics={}
        for arm in ['C025','B025','BASE']:
            tags=['BASE'] if arm=='BASE' else [f'{arm}_seed{s}' for s in config['reference_mask_seeds']]
            preds=[np.load(refroot/'predictions'/f'{t}__test.npz') for t in tags]
            for p in preds:assert np.array_equal(p['y'],yt) and np.array_equal(p['row_indices'],np.arange(len(yt)))
            mean={c:np.mean([p[c] for p in preds],axis=0) for c in conditions};reference_predictions[arm]=mean
            reference_metrics[arm]=grouped({c:average_precision_score(yt,mean[c]) for c in conditions})
        summaries=[]
        for rep in config['randomization_repeats']:
            dest=root/f'rep{rep}';dest.mkdir();(dest/'models').mkdir();(dest/'predictions').mkdir();allpred=[];records=[]
            for rs in config['reference_mask_seeds']:
                maskpath=HERE/'feasibility'/f'split{split}'/f'reference{rs}'/f'rep{rep}'/'mask.npz'
                mask=np.load(maskpath)['new_mask'];xm=xx.copy();xm[mask]=np.nan;train=matrix(xm,med)
                trainsha=arrsha(train);t=time.perf_counter();model=HistGradientBoostingClassifier(**config['model_params'])
                model.fit(train,yy);fitsecs=time.perf_counter()-t;fits+=1
                modelpath=dest/'models'/f'A5_reference{rs}.joblib'
                payload={'model':model,'medians':med,'train_rows':tr,'arm':'A5','alpha':.25,'reference_seed':rs,'randomization_repeat':rep,'model_seed':2026,'split':split,'channel':'M3','config_sha256':sha(HERE/'config.json'),'training_matrix_sha256':trainsha,'training_labels_sha256':arrsha(yy),'mask_sha256':arrsha(mask)}
                joblib.dump(payload,modelpath,compress=3);del train,xm
                predictions={}
                for cond in conditions:
                    xxev=xt.copy();xxev[ev[cond]]=np.nan
                    predictions[cond]=model.predict_proba(matrix(xxev,med))[:,1]
                np.savez_compressed(dest/'predictions'/f'A5_reference{rs}__test.npz',row_indices=np.arange(len(yt)),y=yt,**predictions);allpred.append(predictions)
                records.append({'reference_seed':rs,'fit_seconds':fitsecs,'n_iter':int(model.n_iter_),'effective_params':model.get_params(),'training_matrix_sha256':trainsha,'model_sha256':sha(modelpath),'mask_sha256':arrsha(mask),'probability_sha256':{c:arrsha(p) for c,p in predictions.items()}})
                dump(dest/'fit_records.json',records);log({'event':'fit','split':split,'repeat':rep,'reference_seed':rs,'fits':fits,'seconds':time.perf_counter()-start})
                if time.perf_counter()-start>config['max_run_seconds']:raise TimeoutError('fixed 7200 second split timeout')
            mean={c:np.mean([p[c] for p in allpred],axis=0) for c in conditions}
            np.savez_compressed(dest/'ensemble_predictions.npz',y=yt,row_indices=np.arange(len(yt)),**mean)
            ap={c:float(average_precision_score(yt,p)) for c,p in mean.items()};ga=grouped(ap)
            out={'split':split,'repeat':rep,'ap_per_condition':ap,'ap':ga,'reference_ap':reference_metrics,'contrasts':{'C025-A5':{g:reference_metrics['C025'][g]-ga[g] for g in GROUPS},'A5-B025':{g:ga[g]-reference_metrics['B025'][g] for g in GROUPS},'A5-BASE':{g:ga[g]-reference_metrics['BASE'][g] for g in GROUPS}}}
            dump(dest/'metrics.json',out);summaries.append(out)
        dump(root/'summary.json',{'status':'complete','split':split,'fits':fits,'rows':summaries,'seconds':time.perf_counter()-start,'environment':env(),'freeze_sha256':sha(HERE/'freeze.json')})
        log({'event':'complete','fits':fits})
    except BaseException:
        dump(root/'failure.json',{'timestamp':stamp(),'traceback':traceback.format_exc(),'fits_completed':fits});raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--compile',action='store_true');ap.add_argument('--pilot',action='store_true');ap.add_argument('--feasibility',action='store_true');ap.add_argument('--freeze',action='store_true');ap.add_argument('--check',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--split',type=int);a=ap.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:prepare()
        elif a.compile:compile_lib();dump(HERE/'environment.json',env())
        elif a.pilot:feasibility(True)
        elif a.feasibility:feasibility()
        elif a.freeze:freeze()
        elif a.check:check()
        elif a.run:run(a.split)
        else:ap.error('choose operation')
if __name__=='__main__':main()
