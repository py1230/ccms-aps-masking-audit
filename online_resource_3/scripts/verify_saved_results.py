"""Recompute packaged conditional statistics. No fitting or private paths required."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def compare(x,y):
    error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
    if error>1e-12: raise AssertionError(f'Numeric discrepancy {error}')
    return error

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'saved_result_verification.json')
    args=parser.parse_args();error=0.;checks=0
    # Fingerprints establish local consistency, not historical training provenance.
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for row in manifest:
        p=ROOT/row['destination']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
    base=ROOT/'frozen/original_or3'
    s=json.loads((base/'CCMS_20k_bootstrap_summary.json').read_text())
    matrices=np.load(base/'aps_threeway_draws_20k.npz',allow_pickle=False)
    vectors=[];points=[];bonf=[]
    for role in ['val','test']:
        for family in ['S0','S1_30','S2','S3']:
            for pair in ['C-B','C-M3','B-M3']:
                x=matrices[f'{role}__{family}__{pair}']
                assert x.shape==(20000,5)
                error=max(error,compare(np.quantile(x.mean(1),[.025,.975]),s['results'][role][family][pair]['ci95']));checks+=1
            cb=matrices[f'{role}__{family}__C-B'].mean(1)
            error=max(error,compare(np.quantile(cb,[.025/8,1-.025/8]),s['bonferroni_ci'][role+'_'+family]));checks+=1
            vectors.append(cb);points.append(s['results'][role][family]['C-B']['mean'])
            # Three-way contrasts use the same bootstrap weights.
            error=max(error,compare(matrices[f'{role}__{family}__C-B'],matrices[f'{role}__{family}__C-M3']-matrices[f'{role}__{family}__B-M3']))
    mat=np.array(vectors).T;pts=np.array(points);sd=mat.std(0,ddof=1)
    p=(1+np.sum(np.abs(mat-pts)>=np.abs(pts),axis=0))/(len(mat)+1)
    order=np.argsort(p);adjusted=np.empty(8);adjusted[order]=np.minimum(1,np.maximum.accumulate((8-np.arange(8))*p[order]))
    error=max(error,compare(p,s['holm_p']),compare(adjusted,s['holm_adjusted_p']))
    critical=np.quantile(np.max(np.abs((mat-pts)/sd),axis=1),.975)
    error=max(error,compare(critical,s['sup_t_c_975']))
    for i,key in enumerate([r+'_'+f for r in ['val','test'] for f in ['S0','S1_30','S2','S3']]):
        error=max(error,compare([pts[i]-critical*sd[i],pts[i]+critical*sd[i]],s['sup_t_ci'][key]))
    # Verify all archived strict and legacy recall intervals, not just the eight displayed cells.
    ref=ROOT/'reference/strict_alert';strict=json.loads((ref/'strict_alert_bootstrap_results.json').read_text())
    overall=0;splits=0;weight_checks=0
    for role in ['val','test']:
        z=np.load(ref/f'{role}_paired_recall_draws.npz',allow_pickle=False)
        weights=np.load(ref/f'{role}_shared_row_weight_matrix.npz',allow_pickle=False)
        w=weights['weights'].astype(np.int64);y=weights['unique_labels']
        assert np.all(w.sum(1)==len(y));assert np.all(w[:,y==1].sum(1)==int(y.sum()));weight_checks+=3
        for row in strict['datasets'][role]['rows']:
            for method in ['legacy','strict']:
                x=z[f'{method}__{row["family"]}__{row["pair"]}__{row["budget"]}'];assert x.shape==(1000,5)
                error=max(error,compare(np.quantile(x.mean(1),[.025,.975]),row[method+'_ci95']))
                for i,sp in enumerate(row['per_split']):
                    error=max(error,compare(np.quantile(x[:,i],[.025,.975]),sp[method+'_ci95']))
            overall+=1;splits+=5
    fixtures=json.loads((ref/'boundary_fixtures.json').read_text())['fixtures']
    for f in fixtures:
        w=np.array(f['weights']);y=np.array(f['labels']);k=f['K']
        if 'scores' in f:
            order=np.argsort(-np.array(f['scores']),kind='stable');assert order.tolist()==f['stable_order'];w=w[order];y=y[order]
        expanded=np.repeat(y,w)[:k];assert len(expanded)==f['alerts']==k and int(expanded.sum())==f['TP']
    result={'status':'passed','training_performed':False,'source_fingerprints':len(manifest),'AP_nominal_intervals':24,'AP_Bonferroni_intervals':8,'Holm_family_size':8,'historical_sup_t_recomputed':True,'sup_t_joint_coverage_established':False,'alert_overall_interval_pairs':overall,'alert_split_interval_pairs':splits,'shared_weight_class_checks':weight_checks,'boundary_fixtures':len(fixtures),'max_abs_numeric_error':error,'environment':{'python':platform.python_version(),'numpy':np.__version__},'scope':'Statistics recomputed from saved matrices; no training provenance or independent training claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
