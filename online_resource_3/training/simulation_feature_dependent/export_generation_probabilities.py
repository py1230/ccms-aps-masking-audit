"""Post-run reconstruction of natural missingness probabilities from frozen artifacts.

These are labeled derived artifacts, not newly generated data or a rerun.
"""
from pathlib import Path
import hashlib,json,numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((HERE/'audit/independent_verification.json').read_text())['status']=='passed'
out=HERE/'analysis/generation_probabilities';out.mkdir(exist_ok=False)
provenance=[]
for seed in range(701,731):
    root=HERE/'run_results'/f'seed{seed}';d=np.load(root/'data.npz');r=json.loads((root/'result.json').read_text());masks=np.load(root/'evaluation_masks.npz')
    payload={'train_rows':d['train_rows'],'val_rows':d['val_rows'],'test_rows':d['test_rows']}
    specs=[('train_natural','train',r['train_missing']),('val_natural','val',r['validation_missing'])]+[(f'target_{k}','test',v) for k,v in r['target_missing'].items()]
    maxerr=0.
    for key,subset,meta in specs:
        rows=d[subset+'_rows'];y=d['y'][rows];z=np.clip(d['X'][rows,19],-4,4);p=np.zeros((len(rows),20));ints=np.asarray(meta['intercept_by_class_column'])
        for c in [0,1]:
            ix=np.flatnonzero(y==c);q=np.asarray(meta['q'+str(c)]);use=np.flatnonzero((q>0)&(q<1))
            p[np.ix_(ix,use)]=1/(1+np.exp(-(ints[c,use][None,:]+z[ix,None])))
            p[np.ix_(ix,np.flatnonzero(q==1))]=1.
            maxerr=max(maxerr,float(np.max(np.abs(p[ix].mean(0)-q))))
        assert np.array_equal(d['U_'+subset]<p,masks[key]) and np.all(p[:,19]==0)
        payload[key]=p
    assert maxerr<=1e-12
    path=out/f'seed{seed}.npz';np.savez_compressed(path,**payload)
    provenance.append({'seed':seed,'derived_file':str(path.relative_to(HERE)),'sha256':sha(path),'input_data_sha256':sha(root/'data.npz'),'input_result_sha256':sha(root/'result.json'),'threshold_masks_match':True,'maximum_class_mean_error':maxerr})
(HERE/'analysis/generation_probability_provenance.json').write_text(json.dumps({'scope':'Post-run reconstruction from saved intercepts, class probabilities and complete X; no training or new random draws','script_sha256':sha(Path(__file__)),'all_natural_masks_verified':True,'seeds':provenance},ensure_ascii=False,indent=2)+'\n')
print('Saved and threshold-verified all natural generation probability matrices for 30 seeds')
