"""Independent A5 verification. Does not import training or sampler code.

Checks saved trajectories, forbidden cells, exact margins, raw-data preprocessing,
model inputs, saved model predictions, AP and descriptive aggregation. Reloading
models is an artifact consistency check, not an independent retraining.
"""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'2')
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib, sklearn
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
CONDITIONS=['S0']+[f'S1_{rate}_{s}' for rate in [10,30] for s in [101,202,303]]+[f'S2_{s}' for s in [1101,1202,1303]]+[f'S3_{s}' for s in [2101,2202,2303]]
GROUPS=['S0','S1_10','S1_30','S2','S3']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def raw(path):
    with Path(path).open() as f:
        for skip,line in enumerate(f):
            if line.strip().startswith('class,'):break
        else:raise ValueError('No CSV header')
    df=pd.read_csv(path,skiprows=skip,na_values=['na'],low_memory=False)
    y=df.pop('class');assert y.isin(['pos','neg']).all()
    x=df.to_numpy(dtype=np.float64);assert np.isfinite(x[~np.isnan(x)]).all()
    return x,(y.to_numpy()=='pos').astype(np.int64),list(df.columns)
def ap(y,p):
    y=np.asarray(y);p=np.asarray(p);assert y.shape==p.shape and np.isfinite(p).all()
    order=np.argsort(-p,kind='stable');s=p[order];t=y[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(t,dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/(ends+1)))
def medians(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);m=np.nanmedian(x,axis=0)
    return np.where(np.isfinite(m),m,0.)
def matrix(x,med):
    missing=np.isnan(x)
    return np.column_stack([np.where(missing,med,x),missing.astype(float)])
def evalmask(x,c):
    visible=~np.isnan(x);mask=np.zeros(x.shape,dtype=bool)
    if c=='S0':return mask
    parts=c.split('_');rng=np.random.default_rng(int(parts[-1]))
    if parts[0]=='S1':return visible&(rng.random(x.shape)<int(parts[1])/100)
    if parts[0]=='S2':
        rows=rng.choice(len(x),size=int(round(.5*len(x))),replace=False)
        mask[rows]=visible[rows]&(rng.random((len(rows),x.shape[1]))<.6)
    elif parts[0]=='S3':
        cols=rng.choice(x.shape[1],size=max(1,int(round(.3*x.shape[1]))),replace=False)
        mask[:,cols]=visible[:,cols]
    else:raise ValueError(c)
    return mask
def cmask(x,y,seed):
    observed=~np.isnan(x);q=np.stack([(~observed[y==c]).mean(axis=0) for c in [0,1]])
    rate=np.zeros_like(q);np.divide(.25*(q.max(axis=0)-q),1-q,out=rate,where=q<1)
    return observed&(np.random.default_rng(seed).random(x.shape)<np.clip(rate,0,1)[y])
def close(a,b,tol=1e-12):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta=float(np.max(np.abs(a-b))) if a.size else 0.
    assert delta<=tol,(delta,tol)
    return delta

def grouped(values):
    return {g:float(values['S0']) if g=='S0' else float(np.mean([v for c,v in values.items() if c.startswith(g+'_')])) for g in GROUPS}

# Schema-specific replay, model and summary checks are appended after the runner is reviewed.

def arrsha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()

def replay_lib():
    import ctypes,subprocess
    target=HERE/'verify_switch.dylib'
    subprocess.run(['cc','-O2','-shared','-fPIC',str(HERE/'verify_switch.c'),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target));p=ctypes.c_void_p
    lib.verify_proposals.argtypes=[p,p,p,p,p,p,p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,p,ctypes.c_uint32]
    lib.verify_proposals.restype=ctypes.c_int
    return lib

def replay(path,ref,forbidden,config,lib):
    meta=read(path/'chain.json');trace=loadnp(path/'trace.npz');out=loadnp(path/'mask.npz')
    rr,cc=np.nonzero(ref);rows=rr.astype(np.uint32);cols=cc.astype(np.uint32);edgecount=len(rr)
    assert np.array_equal(rows,trace['initial_edge_rows']) and np.array_equal(cols,trace['initial_edge_cols'])
    assert edgecount==meta['edge_count'] and meta['attempts']==config['attempts_per_edge']*edgecount
    seed=meta['seed'];rng=np.random.default_rng(seed);assert rng.bit_generator.state==meta['rng_initial']
    state=np.array(ref,dtype=np.uint8,order='C');forbidden=np.ascontiguousarray(forbidden,dtype=np.uint8)
    events=trace['accepted_trace'];assert events.dtype==np.uint32 and events.ndim==2 and events.shape[1]==3
    assert len(events)==meta['accepted'] and (np.diff(events[:,0].astype(np.int64))>0).all()
    cursor=0;attempt=0;checkpoints={d['attempt']:d for d in meta['diagnostics']}
    def diag_check():
        d=checkpoints[attempt];m=state.astype(bool)
        assert d['accepted']==cursor and d['mask_sha256']==arrsha(m)
        assert d['hamming']==int(np.count_nonzero(m!=ref))
        assert d['common_reference_edges']==int(np.count_nonzero(m&ref))
        close(d['relocated_fraction'],1-np.count_nonzero(m&ref)/edgecount)
        assert d['forbidden_count']==0 and not np.any(m&forbidden.astype(bool))
        assert np.array_equal(m.sum(axis=0),ref.sum(axis=0)) and np.array_equal(m.sum(axis=1),ref.sum(axis=1))
        dc=np.flatnonzero(ref.sum(0))[:16];assert dc.tolist()==d['diagnostic_cols']
        sm=m[:,dc].astype(np.int64);co=sm.T@sm
        assert co[np.triu_indices(len(dc),1)].tolist()==d['cooccurrence_upper']
    diag_check()
    for batch in meta['batches']:
        assert batch['start']==attempt and rng.bit_generator.state==batch['rng_before']
        n=batch['n'];a=rng.integers(0,edgecount,n,dtype=np.uint32);b=rng.integers(0,edgecount,n,dtype=np.uint32);lazy=rng.integers(0,2,n,dtype=np.uint8)
        assert rng.bit_generator.state==batch['rng_after']
        assert arrsha(a)==batch['a_sha256'] and arrsha(b)==batch['b_sha256'] and arrsha(lazy)==batch['lazy_sha256']
        nextcursor=int(np.searchsorted(events[:,0],attempt+n));segment=np.ascontiguousarray(events[cursor:nextcursor])
        assert len(segment)==batch['accepted']
        code=lib.verify_proposals(state.ctypes.data,forbidden.ctypes.data,rows.ctypes.data,cols.ctypes.data,a.ctypes.data,b.ctypes.data,lazy.ctypes.data,n,state.shape[1],attempt,segment.ctypes.data,len(segment))
        assert code==0,(str(path),attempt,code)
        cursor=nextcursor;attempt+=n
        if attempt in checkpoints:diag_check()
    assert attempt==meta['attempts'] and cursor==len(events) and rng.bit_generator.state==meta['rng_final']
    mask=out['new_mask'];assert mask.dtype==np.bool_ and np.array_equal(state.astype(bool),mask)
    assert np.array_equal(mask.sum(1),out['row_deleted']) and np.array_equal(mask.sum(0),out['col_deleted'])
    assert meta['mask_sha256']==arrsha(mask) and meta['config_sha256']==sha(HERE/'config.json')
    assert np.count_nonzero(mask!=ref)>0
    return mask,{'seed':seed,'attempts_verified':attempt,'accepted_verified':cursor,'relocated_fraction':checkpoints[attempt]['relocated_fraction']}

def check_certificate(path,ref,forbidden):
    rect=loadnp(path)['rectangles'];used=set()
    for r,s,c,d in rect:
        r,s,c,d=map(int,[r,s,c,d]);assert r!=s and c!=d
        assert r not in used and s not in used;used.update([r,s])
        assert ref[r,c] and ref[s,d] and not ref[r,d] and not ref[s,c]
        assert not forbidden[r,d] and not forbidden[s,c]
    assert len(rect)>=100
    return len(rect)

def mask_checks(config,x,y,indices):
    lib=replay_lib();results=[];total=0;certs=[]
    for split in config['split_seeds']:
        src=HERE/'reference'/f'split{split}';pr=loadnp(src/'preprocessing.npz');tr=pr['train_rows']
        assert np.array_equal(tr,indices[f's{split}_train']);xx=x[tr];yy=y[tr];forbid=np.isnan(xx)
        assert np.array_equal(medians(xx),pr['medians'])
        for rs in config['reference_mask_seeds']:
            ref=loadnp(src/'training_masks'/f'C025_seed{rs}.npz')['new_mask']
            assert np.array_equal(ref,cmask(xx,yy,rs)) and not np.any(ref&forbid)
            dest=HERE/'feasibility'/f'split{split}'/f'reference{rs}'
            cert=check_certificate(dest/'legal_switch_certificate.npz',ref,forbid);certs.append(cert)
            masks=[]
            for rep in config['randomization_repeats']:
                path=dest/f'rep{rep}';mask,row=replay(path,ref,forbid,config,lib)
                assert row['seed']==810000000+split*100000+rs*10+rep
                assert row['relocated_fraction']>=config['feasibility_gate']['minimum_reference_edge_relocated_fraction']
                masks.append(mask);total+=row['attempts_verified'];results.append({'split':split,'reference_seed':rs,'repeat':rep,**row})
                print(json.dumps({'event':'mask_verified','split':split,'reference_seed':rs,'repeat':rep}),flush=True)
            div=read(dest/'diversity.json');distances=[int(np.count_nonzero(masks[i]!=masks[j])) for i in range(len(masks)) for j in range(i)]
            assert div['pairwise_hamming']==distances and all(v>0 for v in distances)
            assert div['distinct_masks']==len(masks)==len({arrsha(m) for m in masks})
            assert div['legal_space_lower_bound']==f'2^{cert}'
    assert len(results)==75
    return {'status':'passed','chains':len(results),'all_proposals_replayed':total,'certificates':certs,'rows':results}

def reference_masks(x,y,seed,arm):
    if arm=='BASE':return np.zeros(x.shape,dtype=bool)
    if arm=='C025':return cmask(x,y,seed)
    obs=~np.isnan(x);q=np.stack([(~obs[y==c]).mean(0) for c in [0,1]])
    weights=np.bincount(y,minlength=2)/len(y);mean=weights@q
    p=np.zeros(x.shape[1]);np.divide(.25*(q.max(0)-mean),1-mean,out=p,where=mean<1)
    return obs&(np.random.default_rng(seed).random(x.shape)<np.clip(p,0,1))

def model_checks(config,x,y,xt,yt,indices):
    maxprob=0.;maxap=0.;fits=0;reference_fits=0;arrays=0;rows=[]
    emasks=loadnp(HERE/'reference/test_evaluation_masks.npz');assert set(emasks)==set(CONDITIONS)
    for c in CONDITIONS:assert np.array_equal(emasks[c],evalmask(xt,c))
    for split in config['split_seeds']:
        source=HERE/'reference'/f'split{split}';pr=loadnp(source/'preprocessing.npz');tr=pr['train_rows'];med=pr['medians'];xx=x[tr];yy=y[tr]
        assert np.array_equal(tr,indices[f's{split}_train']) and np.array_equal(med,medians(xx))
        def predict_check(model_path,pred_path,reference_arm=None,reference_seed=None,rep=None):
            nonlocal maxprob,arrays
            obj=joblib.load(model_path);model=obj['model'];saved=loadnp(pred_path)
            assert model.get_params()==config['model_params'] and model.n_features_in_==340
            assert np.array_equal(obj['medians'],med) and np.array_equal(obj['train_rows'],tr)
            assert obj['channel']=='M3' and obj['model_seed']==2026
            assert np.array_equal(saved['y'],yt) and np.array_equal(saved['row_indices'],np.arange(len(yt)))
            assert set(saved)==set(CONDITIONS+['y','row_indices'])
            if reference_arm is not None:
                tag='BASE' if reference_arm=='BASE' else f'{reference_arm}_seed{reference_seed}'
                mask=loadnp(source/'training_masks'/f'{tag}.npz')['new_mask']
                assert np.array_equal(mask,reference_masks(xx,yy,reference_seed,reference_arm))
                assert obj['repeat']==1
            else:
                mask=loadnp(HERE/'feasibility'/f'split{split}'/f'reference{reference_seed}'/f'rep{rep}'/'mask.npz')['new_mask']
                xm=xx.copy();xm[mask]=np.nan;train=matrix(xm,med)
                assert obj['training_matrix_sha256']==arrsha(train) and obj['training_labels_sha256']==arrsha(yy)
                assert obj['mask_sha256']==arrsha(mask) and obj['config_sha256']==sha(HERE/'config.json')
                assert obj['split']==split and obj['randomization_repeat']==rep and obj['reference_seed']==reference_seed and obj['arm']=='A5' and obj['alpha']==.25
                del xm,train
            for c in CONDITIONS:
                ev=xt.copy();ev[emasks[c]]=np.nan;pred=model.predict_proba(matrix(ev,med))[:,1]
                maxprob=max(maxprob,close(saved[c],pred));arrays+=1
            return {c:saved[c] for c in CONDITIONS},obj
        refs={}
        for arm in ['BASE','C025','B025']:
            seeds=[None] if arm=='BASE' else config['reference_mask_seeds'];pred=[]
            for rs in seeds:
                tag='BASE' if arm=='BASE' else f'{arm}_seed{rs}'
                ps,obj=predict_check(source/'models'/f'{tag}.joblib',source/'predictions'/f'{tag}__test.npz',arm,rs)
                pred.append(ps);reference_fits+=1
            mean={c:np.mean([p[c] for p in pred],axis=0) for c in CONDITIONS}
            refs[arm]=grouped({c:ap(yt,mean[c]) for c in CONDITIONS})
        summ=read(HERE/'run_results'/f'split{split}'/'summary.json')
        assert summ['status']=='complete' and summ['fits']==15 and not (HERE/'run_results'/f'split{split}'/'failure.json').exists()
        for rep in config['randomization_repeats']:
            dest=HERE/'run_results'/f'split{split}'/f'rep{rep}';record=read(dest/'fit_records.json');pred=[]
            assert len(record)==3
            for rs,rec in zip(config['reference_mask_seeds'],record):
                modelpath=dest/'models'/f'A5_reference{rs}.joblib'
                ps,obj=predict_check(modelpath,dest/'predictions'/f'A5_reference{rs}__test.npz',None,rs,rep);pred.append(ps);fits+=1
                assert rec['reference_seed']==rs and rec['model_sha256']==sha(modelpath) and rec['effective_params']==config['model_params']
                assert rec['training_matrix_sha256']==obj['training_matrix_sha256'] and rec['mask_sha256']==obj['mask_sha256']
                assert rec['n_iter']==int(obj['model'].n_iter_)
                for c in CONDITIONS:assert rec['probability_sha256'][c]==arrsha(ps[c])
            ens=loadnp(dest/'ensemble_predictions.npz');metrics=read(dest/'metrics.json')
            assert np.array_equal(ens['y'],yt) and np.array_equal(ens['row_indices'],np.arange(len(yt)))
            aps={}
            for c in CONDITIONS:
                prob=np.mean([p[c] for p in pred],axis=0);maxprob=max(maxprob,close(ens[c],prob));aps[c]=ap(yt,prob)
                maxap=max(maxap,close(aps[c],metrics['ap_per_condition'][c]))
            aps=grouped(aps);contrasts={key:{} for key in ['C025-A5','A5-B025','A5-BASE']}
            for g in GROUPS:
                maxap=max(maxap,close(aps[g],metrics['ap'][g]))
                for arm in refs:maxap=max(maxap,close(refs[arm][g],metrics['reference_ap'][arm][g]))
                for key,value in [('C025-A5',refs['C025'][g]-aps[g]),('A5-B025',aps[g]-refs['B025'][g]),('A5-BASE',aps[g]-refs['BASE'][g])]:
                    contrasts[key][g]=value;maxap=max(maxap,close(value,metrics['contrasts'][key][g]))
            assert metrics==next(r for r in summ['rows'] if r['repeat']==rep)
            rows.append({'split':split,'repeat':rep,'ap':aps,'reference_ap':refs,'contrasts':contrasts})
            print(json.dumps({'event':'models_verified','split':split,'repeat':rep,'fits':fits}),flush=True)
    assert fits==75 and reference_fits==35 and arrays==1430
    repeated=[]
    for rep in config['randomization_repeats']:
        unit=[r for r in rows if r['repeat']==rep];assert len(unit)==5
        repeated.append({'repeat':rep,'contrasts':{c:{g:float(np.mean([r['contrasts'][c][g] for r in unit])) for g in GROUPS} for c in ['C025-A5','A5-B025','A5-BASE']}})
    desc={c:{g:{'mean':float(np.mean([r['contrasts'][c][g] for r in repeated])),'min':float(np.min([r['contrasts'][c][g] for r in repeated])),'max':float(np.max([r['contrasts'][c][g] for r in repeated]))} for g in GROUPS} for c in ['C025-A5','A5-B025','A5-BASE']}
    return {'status':'passed','a5_models':fits,'reference_models':reference_fits,'probability_arrays':arrays,'ensembles':25,'ap_per_condition_checks':325,'max_prediction_error':maxprob,'max_ap_error':maxap,'units':rows,'per_repeat':repeated,'descriptive':desc}

def structure_checks(config,x,y):
    import csv
    saved=read(HERE/'mask_structure_summary.json');assert saved['chains']==75 and saved['class_total_deleted_exactly_preserved'] is True
    checks=0;computed=[]
    with (HERE/'mask_structure_summary.csv').open() as f:csvrows=list(csv.DictReader(f))
    for row in saved['rows']:
        split,rs,rep=[row[k] for k in ['split','reference_seed','repeat']]
        root=HERE/'reference'/f'split{split}';tr=loadnp(root/'preprocessing.npz')['train_rows'];labels=y[tr]
        ref=loadnp(root/'training_masks'/f'C025_seed{rs}.npz')['new_mask']
        dest=HERE/'feasibility'/f'split{split}'/f'reference{rs}'/f'rep{rep}'
        mask=loadnp(dest/'mask.npz')['new_mask'];meta=read(dest/'chain.json')
        diag=loadnp(HERE/'structure_diagnostics'/f'split{split}_reference{rs}_rep{rep}.npz')
        newclass=np.zeros((2,170),dtype=np.int64);oldclass=np.zeros_like(newclass)
        np.add.at(newclass,labels,mask.astype(np.int64));np.add.at(oldclass,labels,ref.astype(np.int64))
        assert np.array_equal(newclass,diag['class_col_deleted_A5']) and np.array_equal(oldclass,diag['class_col_deleted_C'])
        assert np.array_equal(diag['row_class'],labels) and np.array_equal(diag['row_deleted_C'],ref.sum(1)) and np.array_equal(diag['row_deleted_A5'],mask.sum(1))
        assert np.array_equal(newclass.sum(1),oldclass.sum(1))
        diff=newclass-oldclass;changed=(mask!=ref)
        c0=np.asarray(meta['diagnostics'][0]['cooccurrence_upper']);c10=np.asarray(meta['diagnostics'][-2]['cooccurrence_upper']);c20=np.asarray(meta['diagnostics'][-1]['cooccurrence_upper'])
        values={'changed_rows':int(np.sum(np.sum(changed,1)>0)),'changed_row_fraction':float(np.mean(np.sum(changed,1)>0)),'changed_cols':int(np.sum(np.sum(changed,0)>0)),'changed_col_fraction':float(np.mean(np.sum(changed,0)>0)),'class_col_l1_change':int(np.sum(np.abs(diff))),'class_col_max_change':int(np.max(np.abs(diff))),'class_col_nonzero_changes':int(np.count_nonzero(diff)),'cooccurrence16_l1_from_C':int(np.sum(np.abs(c20-c0))),'cooccurrence16_l1_10E_to_20E':int(np.sum(np.abs(c20-c10))),'cooccurrence16_changed_pairs_10E_to_20E':int(np.count_nonzero(c20-c10)),'relocated_fraction':1-np.sum(mask&ref)/np.sum(ref)}
        for c in [0,1]:
            values[f'class{c}_total_deleted_C']=int(oldclass[c].sum());values[f'class{c}_total_deleted_A5']=int(newclass[c].sum())
            visible_counts=np.sum(~np.isnan(x[tr][labels==c]),axis=0)
            assert np.array_equal(visible_counts,diag['class_col_visible'][c])
            values[f'class{c}_visible_cells']=int(visible_counts.sum())
            values[f'class{c}_deleted_per_visible']=newclass[c].sum()/diag['class_col_visible'][c].sum()
        match=[r for r in csvrows if all(int(r[k])==row[k] for k in ['split','reference_seed','repeat'])];assert len(match)==1
        for k,v in values.items():close(v,row[k]);close(v,float(match[0][k]));checks+=2
        computed.append(values)
    for k,r in saved['ranges'].items():
        vals=[v[k] for v in computed]
        for name,value in [('min',min(vals)),('max',max(vals)),('mean',float(np.mean(vals)))]:close(r[name],value);checks+=1
    return {'status':'passed','chains':75,'checks':checks,'class_totals_exact':True,'source_sha256':sha(HERE/'mask_structure_summary.json'),'class_col_l1_range':saved['ranges']['class_col_l1_change'],'cooccurrence16_10E_20E_range':saved['ranges']['cooccurrence16_l1_10E_to_20E'],'scope':'post hoc structural diagnostics; checkpoint cooccurrence counts independently replayed in mask verification; no mixing claim'}

def summary_checks(models,config):
    import csv
    units=[]
    measures=['A5','C025','B025','BASE','C025-A5','A5-B025','A5-BASE']
    for row in models['units']:
        for g in GROUPS:
            units.append({'split':row['split'],'repeat':row['repeat'],'group':g,'A5':row['ap'][g],**{arm:row['reference_ap'][arm][g] for arm in ['C025','B025','BASE']},**{c:row['contrasts'][c][g] for c in ['C025-A5','A5-B025','A5-BASE']}})
    repeats=[]
    for rep in config['randomization_repeats']:
        for g in GROUPS:
            sub=[r for r in units if r['repeat']==rep and r['group']==g]
            repeats.append({'repeat':rep,'group':g,**{m:float(np.sum([r[m] for r in sub])/5) for m in measures}})
    stats=[]
    for g in GROUPS:
        for m in measures:
            vals=[r[m] for r in repeats if r['group']==g]
            stats.append({'group':g,'measure':m,'mean':float(np.sum(vals)/5),'min':min(vals),'max':max(vals),'repeat_values':vals})
    archived=read(HERE/'descriptive_summary.json');assert archived['status']=='complete' and archived['fits']==75 and archived['fixed_splits']==5 and archived['independent_randomization_repeats']==5
    assert len(archived['results'])==len(stats)==35
    errors=[]
    for row in stats:
        match=[r for r in archived['results'] if (r['group'],r['measure'])==(row['group'],row['measure'])];assert len(match)==1
        for key in ['mean','min','max','repeat_values']:errors.append(close(row[key],match[0][key]))
    primary=next(r for r in stats if r['group']=='S1_30' and r['measure']=='C025-A5')
    for key in ['mean','min','max','repeat_values']:errors.append(close(primary[key],archived['primary'][key]))
    def check_csv(name,expected,keys,vals):
        with (HERE/name).open() as f:got=list(csv.DictReader(f))
        assert len(got)==len(expected)
        for row in expected:
            match=[r for r in got if all(str(row[k])==r[k] for k in keys)];assert len(match)==1
            for key in vals:errors.append(close(row[key],float(match[0][key])))
    check_csv('ap_and_contrasts_by_unit.csv',units,['split','repeat','group'],measures)
    check_csv('ap_and_contrasts_by_repeat.csv',repeats,['repeat','group'],measures)
    check_csv('descriptive_summary.csv',stats,['group','measure'],['mean','min','max'])
    return {'status':'passed','json_statistics':35,'csv_unit_rows':125,'csv_repeat_rows':25,'csv_summary_rows':35,'max_error':max(errors),'primary':primary,'file_sha256':{name:sha(HERE/name) for name in ['descriptive_summary.json','ap_and_contrasts_by_unit.csv','ap_and_contrasts_by_repeat.csv','descriptive_summary.csv']}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--masks-only',action='store_true');parser.add_argument('--models-only',action='store_true');args=parser.parse_args()
    start=time.perf_counter();config=read(HERE/'config.json');x,y,features=raw(HERE/'raw_data/aps_failure_training_set.csv');xt,yt,ft=raw(HERE/'raw_data/aps_failure_test_set.csv')
    assert features==ft and x.shape==(60000,170) and xt.shape==(16000,170) and int(y.sum())==1000 and int(yt.sum())==375
    assert config['input_environment']=={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}
    for yy,pp,expected in [([1,0],[.5,.5],.5),([1,0,1,0],[.9,.8,.7,.6],5/6),([0,1,1,0],[.9,.8,.8,.7],2/3)]:assert abs(ap(yy,pp)-expected)<1e-14
    indices=loadnp(HERE/'split_indices.npz')
    for s in config['split_seeds']:
        pieces=[indices[f's{s}_{v}'] for v in ['train','val','reserve']]
        assert [len(p) for p in pieces]==[42000,9000,9000] and np.array_equal(np.sort(np.concatenate(pieces)),np.arange(len(y)))
    with threadpool_limits(limits=2):
        result={'status':'passed','timestamp_utc':datetime.now(timezone.utc).isoformat(),'verifier_sha256':sha(__file__),'verifier_c_sha256':sha(HERE/'verify_switch.c'),'scope':'independent reconstruction and reload, not retraining or historical provenance'}
        if not args.models_only:result['masks']=mask_checks(config,x,y,indices)
        if not args.masks_only:
            fr=read(HERE/'freeze.json')
            for name,h in fr['files'].items():assert sha(HERE/name)==h,name
            result['freeze_sha256']=sha(HERE/'freeze.json')
            result['models']=model_checks(config,x,y,xt,yt,indices)
            result['summary']=summary_checks(result['models'],config)
            result['structure']=structure_checks(config,x,y)
    result['input_config_sha256']=sha(HERE/'config.json')
    result['source_files_checked']={}
    for name,rec in read(HERE/'reference_sources.json').items():
        assert sha(HERE/name)==rec['sha256'],name
        if Path(rec['source']).is_file():assert sha(rec['source'])==rec['sha256'],name
        result['source_files_checked'][name]=rec['sha256']
    result['wall_seconds']=time.perf_counter()-start
    filename='verification_masks.json' if args.masks_only else 'verification_models.json' if args.models_only else 'verification.json'
    save(HERE/filename,result);print(json.dumps({'status':'passed','output':filename,'seconds':result['wall_seconds']}),flush=True)
if __name__=='__main__':main()
