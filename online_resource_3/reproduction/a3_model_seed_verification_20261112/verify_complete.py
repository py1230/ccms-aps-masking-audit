"""Strict saved-model replay. No classifier training; failures return nonzero.

Usage: python verify_complete.py --root RECEIVED_ROOT --out report.json
Requires numpy, pandas, scikit-learn and joblib; imports the reviewed pipeline
module solely to deserialize the supplied MedianImputer objects.
"""
import argparse, csv, hashlib, json, platform, re, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.metrics import average_precision_score

def sha(b): return hashlib.sha256(b).hexdigest()
def fh(p): return sha(p.read_bytes())
def check(ok, message):
    if not bool(ok): raise ValueError(message)
def equal(a,b,message): check(np.array_equal(a,b),message)
def unpack(z,key,shape):
    b=z[key]; n=int(np.prod(shape))
    check(b.dtype==np.uint8 and b.size==(n+7)//8,key+' packed size')
    return np.unpackbits(b)[:n].reshape(shape).astype(bool)
def split(y,s):
    rng=np.random.default_rng(s); p=[[],[],[]]
    for c in np.unique(y):
        idx=np.flatnonzero(y==c);rng.shuffle(idx)
        bounds=[round(v*len(idx)/60000) for v in (0,42000,51000,60000)]
        for k in range(3):p[k].append(idx[bounds[k]:bounds[k+1]])
    return [np.sort(np.concatenate(v)) for v in p]

def verify(root,out):
    t=time.time(); result={'status':'RUNNING','checks':[], 'cells':[], 'splits':[]}
    result['environment']={'python':platform.python_version(),'numpy':np.__version__,
        'pandas':pd.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,
        'platform':platform.platform(),'script_sha256':fh(Path(__file__))}
    try:
        listed=[]
        for line in (root/'SHA256SUMS.txt').read_text().splitlines():
            if not line or line.startswith('#'):continue
            h,rel=line.split(None,1);rel=rel.strip().lstrip('*');p=root/rel
            check(p.resolve().is_relative_to(root.resolve()),'manifest path traversal')
            check(fh(p)==h,'manifest mismatch: '+rel);listed.append(p.resolve())
        check(len(listed)==len(set(listed))==119,'manifest entry count')
        actual={p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        check(actual-set(listed)=={(root/'SHA256SUMS.txt').resolve()},'unlisted payload')
        result['manifest_payloads_verified']=119
        hroot=root/'01_历史原件';run=root/'02_11-12复现产物/A3_repro_2026-11-12';new=root/'03_本次事后重建与核验';reb=new/'rebuilt'
        env=json.loads((reb/'rebuild_environment.json').read_text())
        data=new/'data/aps_failure_training_set.csv'
        check(fh(data)==env['data_sha256'],'raw data hash')
        check(fh(new/'scripts/rebuild_a3_supplement.py')==env['script_sha256'],'rebuild script hash')
        for path,h in env['dependencies'].items():check(fh(new/'code'/Path(path).name)==h,'dependency hash '+path)
        receipt=json.loads((new/'logs/execution_receipt.json').read_text())
        for name,v in receipt['logs'].items():
            p=new/'logs'/name;check(fh(p)==v['sha256'] and p.stat().st_size==v['bytes'],'receipt log hash')
        record=(hroot/'00_历史追溯记录.md').read_text()
        command=re.search(r'```bash\n(.*?)\n```',record,re.S).group(1)
        body=command.split("<<'EOF'\n",1)[1].split('\nEOF',1)[0]+'\n'
        check(body.encode()==(hroot/'运行脚本_run_seed_dispersion.py').read_bytes(),'heredoc runner bytes')
        meta=json.loads((hroot/'harness_meta.json').read_text())
        check(command.startswith(meta['command']),'historical command prefix')
        result['historical_command']={'transcript_characters':len(command),'runner_bytes':len(body.encode()),
            'prefix_characters':len(meta['command']),'heredoc_and_prefix_match':True,
            'raw_chat_projection_included':False,'background_launch':True,
            'exit_code_scope':'launcher shell only; Python training exit status not established'}
        df=pd.read_csv(data,skiprows=20,na_values=['na','NA','Na','NaN','nan','?',''],low_memory=False)
        X=df.drop(columns='class').to_numpy(dtype=float);y=df['class'].map({'neg':0,'pos':1}).to_numpy(dtype=int)
        check(X.shape==(60000,170) and y.sum()==1000,'raw dimensions/labels')
        features=json.loads((reb/'feature_order.json').read_text())
        check(features['order']==list(df.drop(columns='class').columns) and features['source_sha256']==fh(data),'feature order')
        groups={}
        with (run/'a3_persample_predictions.csv').open(newline='') as f:
            for r in csv.DictReader(f):groups.setdefault((int(r['split_seed']),r['arm'],int(r['model_seed'])),[]).append(r)
        expected={(s,a,m) for s in (11,22,33,44,55) for a in ('C','B') for m in (2026,2027,2028)}
        check(set(groups)==expected and sum(map(len,groups.values()))==270000,'CSV groups/rows')
        sys.dont_write_bytecode=True;sys.path.insert(0,str(new/'code'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for s in (11,22,33,44,55):
                tr,va,ca=split(y,s);Xt=X[tr];Xv=X[va];yv=y[va]
                z=np.load(run/f'preds_s{s}.npz',allow_pickle=False)
                em=np.load(reb/f'eval_masks_s{s}.npz',allow_pickle=False)
                for saved in (z,em):equal(saved['val_rows'],va,'validation rows');equal(saved['y_val'],yv,'validation labels')
                equal(em['mask_shape'],Xv.shape,'evaluation shape')
                check(float(em['added_rate_prefix'][0])==.3 and int(em['added_seed_prefix'][0])==101,'eval recipe')
                native=np.isnan(Xv);added=(~native)&(np.random.default_rng(101).random(Xv.shape)<.3);merged=native|added
                for name,a in [('native',native),('added',added),('merged',merged)]:
                    equal(unpack(em,name+'_bits',Xv.shape),a,'evaluation mask '+name)
                    check(int(em['n_'+name])==int(a.sum()),'evaluation count '+name)
                med=np.nanmedian(Xt,axis=0);empty=np.flatnonzero(~np.isfinite(med));med[~np.isfinite(med)]=0
                iz=np.load(reb/f'imputer_s{s}.npz',allow_pickle=False)
                equal(iz['medians'],med,'training-only medians');equal(iz['all_nan_columns'],empty,'allnan columns')
                check(int(iz['fitted_on_rows'][0])==42000 and int(iz['n_features_in'])==170,'imputer dimensions')
                imp=joblib.load(reb/f'imputer_s{s}.joblib')
                equal(imp.medians_,med,'loaded imputer medians');equal(imp.all_nan_columns_,empty,'loaded imputer empty')
                ev=Xv.copy();ev[added]=np.nan
                filled=np.where(merged,med[None,:],ev);equal(imp.transform(ev),filled,'loaded imputer transform')
                M=np.hstack([filled,merged.astype(float)]);mhash=sha(np.ascontiguousarray(M).tobytes())
                check(mhash==str(em['m3_eval_sha256'][0]),'M3 matrix hash')
                row=next(v for v in env['per_split'] if v['split']==s)
                check(mhash==row['m3_eval_sha256'] and sha(med.tobytes())==row['imputer_median_sha256'],'rebuild metadata')
                miss=np.isnan(Xt);yt=y[tr];counts=np.bincount(yt,minlength=2)
                q=np.array([miss[yt==c].sum(axis=0)/counts[c] for c in range(2)]);qmax=q.max(axis=0)
                cr=np.divide(.25*(qmax[None,:]-q),1-q,out=np.zeros_like(q),where=1-q>0)
                qb=(counts/counts.sum())@q;br=np.divide(.25*(qmax-qb),1-qb,out=np.zeros_like(qb),where=1-qb>0)
                u=np.random.default_rng(9107).random(Xt.shape)
                cells=json.loads((run/f'cells_s{s}.json').read_text())['cells']
                equal(z['delmask_shape'],Xt.shape,'training mask shape')
                for ai,arm in enumerate(('C','B')):
                    mask=(~miss)&(u<(cr[yt] if arm=='C' else br[None,:]))
                    equal(unpack(z,'delmask_'+arm,Xt.shape),mask,'training mask')
                    maskhash=sha(np.ascontiguousarray(mask).tobytes());check(maskhash==str(z['mask_sha256'][ai]),'training mask hash')
                    for ms in (2026,2027,2028):
                        rr=sorted(groups[(s,arm,ms)],key=lambda r:int(r['val_index']))
                        equal([int(r['val_index']) for r in rr],np.arange(9000),'all-group val index')
                        equal([int(r['source_row_0based']) for r in rr],va,'all-group source rows')
                        equal([int(r['y_true']) for r in rr],yv,'all-group labels')
                        pc=np.array([float(r['prob']) for r in rr]);pn=z[f'{arm}_{ms}']
                        equal(pc,pn,'CSV/NPZ exact probabilities')
                        check(np.isfinite(pc).all() and ((pc>=0)&(pc<=1)).all(),'probability domain')
                        cell=next(c for c in cells if c['arm']==arm and c['model_seed']==ms)
                        mf=run/f'model_s{s}_{arm}_{ms}.joblib'
                        check(fh(mf)==cell['model_sha256'] and maskhash==cell['mask_sha256'],'model/cell hashes')
                        model=joblib.load(mf);equal(model.classes_,[0,1],'model classes')
                        check(model.n_features_in_==340 and model.random_state==ms,'model metadata')
                        prob=model.predict_proba(M)[:,1]
                        dp=float(np.max(np.abs(prob-pn)));check(np.isfinite(dp) and dp<=1e-12,'model probability replay')
                        hist=json.loads((hroot/f'results_seed/s{s}_{arm}_{ms}.json').read_text())
                        check((hist['split'],hist['arm'],hist['model_seed'])==(s,arm,ms),'history identity')
                        ap=float(average_precision_score(yv,prob));delta=abs(ap-hist['ap']);check(delta<=1e-12,'historical AP agreement')
                        check(cell['ap']==cell['recorded_ap']==hist['ap'],'scalar cross-reference')
                        result['cells'].append({'split':s,'arm':arm,'model_seed':ms,'max_probability_difference':dp,
                            'AP':ap,'historical_AP':hist['ap'],'AP_difference':delta,'all_sample_mapping_verified':True})
                result['splits'].append({'split':s,'evaluation_masks_verified':3,'training_masks_verified':2,
                    'imputer_verified':True,'M3_sha256':mhash,'shape':list(M.shape)})
                print('verified split',s,flush=True)
            result['warnings']=sorted(set(str(w.message) for w in caught))
        result.update(status='PASS_WITH_PROVENANCE_LIMITATIONS',model_inference_replayed=30,training_fits=0,
            prediction_rows=270000,max_probability_difference=max(c['max_probability_difference'] for c in result['cells']),
            max_AP_difference=max(c['AP_difference'] for c in result['cells']))
        result['limits']=['Historical fitted models/masks/probabilities/effective environment remain unavailable.',
            'Evaluation masks and imputers are post hoc reconstructions.',
            'Prediction agreement does not prove uniqueness or identity of unavailable historical inputs.',
            'Historical command is a supplied transcription; raw conversation projection is absent.',
            'This is model replay in the recorded local runtime, not historical training authentication.']
        return_code=0
    except Exception as e:
        result.update(status='FAIL',error=f'{type(e).__name__}: {e}');return_code=1
    result['wall_seconds']=round(time.time()-t,3);result['generated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result.get(k) for k in ('status','error','model_inference_replayed','max_probability_difference','max_AP_difference','wall_seconds')},ensure_ascii=False))
    return return_code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'received');p.add_argument('--out',type=Path,default=Path('a3_replay_verification.json'));a=p.parse_args()
    sys.exit(verify(a.root.resolve(),a.out.resolve()))
