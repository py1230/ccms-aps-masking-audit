"""Stage 4 formal strict-capacity correction; no model training or archive edits.

This script independently reconstructs the paired class-stratified bootstrap,
retains the legacy calculation for exact replay, and verifies every component
against explicitly expanded row occurrences using a separate implementation.
Run with bundled Python. All outputs remain beside this script.
"""
from pathlib import Path
import hashlib, json, math, sys, time
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
HIST = Path('/Users/bgc/Documents/ChatGPT/research/rr/02_原完整复现包/CCMS论文修订完整资料包_2026-11-01_192149/02_本轮代码与结果/results')
LEGACY = Path('/Users/bgc/Documents/ChatGPT/research/new/revision/analysis/r15_decision_metrics.json')
LEGACY_CODE = LEGACY.with_name('run_r15_decision.py')
SEEDS = [11,22,33,44,55]
FAM = {'S0':['S0'],'S1_30':[f'S1_30_{s}' for s in [101,202,303]],'S2':[f'S2_{s}' for s in [1101,1202,1303]],'S3':[f'S3_{s}' for s in [2101,2202,2303]]}
ARMS = ['M3','C025','B025']
RHOS = ['rho_0.02','rho_prevalence']
PAIRS = {'C-B':(1,2), 'C-M3':(1,0), 'B-M3':(2,0)}
B = 1000

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(a): return np.quantile(a, [.025,.975]).tolist()
def exact_budget(n, numerator, denominator):
    return max(1,min(n,(int(n)*int(numerator)+int(denominator)-1)//int(denominator)))

def weighted_strict(w, y, K):
    """Truncate copies of the boundary row so selected count is exactly K."""
    cw=np.cumsum(w); cp=np.cumsum(w*y)
    j=int(np.searchsorted(cw,K,side='left'))
    excess=int(cw[j])-K
    tp=int(cp[j])-excess*int(y[j])
    return tp, int(cw[j])-excess, cw, cp

def expanded_strict(w, y, K):
    """Independent occurrence-space oracle: materialize, take first K, count."""
    expanded=np.repeat(y,w)
    top=expanded[:K]
    return int(top.sum()),len(top),int(expanded.sum())

def main():
    started=time.perf_counter()
    inputs=[HIST/f'aps_seed{s}.npz' for s in SEEDS]+[LEGACY,LEGACY_CODE]
    manifest=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs]
    archive=json.loads(LEGACY.read_text())
    packs=[np.load(p,allow_pickle=False) for p in inputs[:5]]
    all_results={}; allrows=[]; saved=[]
    for dataset in ['val','test']:
        ys=[z['y_'+dataset].astype(np.int64) for z in packs]
        ids=[z['val_rows'] for z in packs] if dataset=='val' else [np.arange(len(y)) for y in ys]
        union=np.unique(np.concatenate(ids)); maps=[np.searchsorted(union,x) for x in ids]
        uy=np.full(len(union),-1,dtype=np.int8)
        for y,m in zip(ys,maps):
            assert np.all((uy[m]==-1)|(uy[m]==y)), 'Shared original row has inconsistent label'
            uy[m]=y
        assert np.all(uy>=0)
        rng=np.random.default_rng(20261101)
        pos=np.flatnonzero(uy==1); neg=np.flatnonzero(uy==0)
        weight_matrix=np.zeros((B,len(union)),dtype=np.uint8)
        for b in range(B):
            sampled=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            w=np.bincount(sampled,minlength=len(union))
            assert w.max()<256
            weight_matrix[b]=w
        assert np.all(weight_matrix.sum(1)==len(union))
        assert np.all(weight_matrix[:,pos].sum(1)==len(pos))
        assert np.all(weight_matrix[:,neg].sum(1)==len(neg))
        weights_path=OUT/f'{dataset}_shared_row_weight_matrix.npz'
        np.savez_compressed(weights_path,weights=weight_matrix,unique_row_ids=union,unique_labels=uy,
                            **{f'split_{SEEDS[i]}_mapping':m for i,m in enumerate(maps)})
        saved.append(weights_path)
        components=[]; point_cache={}; legacy_draws={}; strict_draws={}
        cache={}; ties=[]; point_checks=0; max_point_error=0.
        for f,conds in FAM.items():
            for si,(y,z) in enumerate(zip(ys,packs)):
                for ai,arm in enumerate(ARMS):
                    replica_points=[]
                    for ci_,condition in enumerate(conds):
                        p=z[f'{arm}__{dataset}__{condition}']
                        assert len(p)==len(y) and np.isfinite(p).all()
                        order=np.argsort(-p,kind='stable')
                        # Separate ordering implementation makes original stored row position explicit.
                        oracle_order=np.lexsort((np.arange(len(p)),-p))
                        assert np.array_equal(order,oracle_order)
                        yo=y[order]; cache[f,si,ci_,ai]=(order,yo,oracle_order,y[oracle_order])
                        values={}
                        for rate in [.005,.01,.02,.05,.10]:
                            K=exact_budget(len(y),int(round(rate*1000)),1000)
                            values[str(rate)]=float(yo[:K].sum()/y.sum())
                            if K<len(y) and p[order[K-1]]==p[order[K]]:
                                ties.append({'family':f,'seed':SEEDS[si],'arm':arm,'condition':condition,'rho':rate,'K':K})
                        values['rho_prevalence']=float(yo[:int(y.sum())].sum()/y.sum())
                        replica_points.append(values)
                    for key in replica_points[0]:
                        point_cache[f,si,ai,key]=float(np.mean([v[key] for v in replica_points]))
                    old=archive['results'][dataset][f][arm]['per_split'][si]
                    for rate in [.005,.01,.02,.05,.10]:
                        delta=abs(point_cache[f,si,ai,str(rate)]-old['recall_at'][f'{rate:g}'])
                        max_point_error=max(max_point_error,delta); point_checks+=1; assert delta<1e-12
                    assert abs(point_cache[f,si,ai,'rho_prevalence']-old['recall_prevalence'])<1e-12
            for pair in PAIRS:
                for rh in RHOS:
                    legacy_draws[f,pair,rh]=np.zeros((B,5));strict_draws[f,pair,rh]=np.zeros((B,5))
        checks={'components_verified':0,'capacity_mismatches':0,'integer_TP_mismatches':0,'denominator_mismatches':0,
                'legacy_over_budget_components':0,'max_legacy_excess':0,'legacy_recall_changed_components':0,
                'rho_0.02_C_B_components':0,'rho_0.02_C_B_over_budget':0,'max_point_recall_error':max_point_error,
                'point_recall_cells_checked':point_checks+60,'all_tie_orders_equal':True}
        for b in range(B):
            shared=weight_matrix[b].astype(np.int64)
            for si,(m,y) in enumerate(zip(maps,ys)):
                w=shared[m]; n=int(w.sum()); P=int(np.dot(w,y))
                assert P>0
                for f,conds in FAM.items():
                    orig=np.zeros((len(conds),3,2)); fixed=np.zeros_like(orig)
                    for ci_,condition in enumerate(conds):
                        for ai,arm in enumerate(ARMS):
                            order,yo,oracle_order,oracle_y=cache[f,si,ci_,ai]
                            ww=w[order]
                            # Build cumulative arrays once; two budgets share the same occurrence population.
                            cw=np.cumsum(ww);cp=np.cumsum(ww*yo)
                            for ri,rh in enumerate(RHOS):
                                num,den=(1,50) if ri==0 else (int(y.sum()),len(y))
                                K=exact_budget(n,num,den)
                                j=int(np.searchsorted(cw,K,side='left')); excess=int(cw[j])-K
                                tp=int(cp[j])-excess*int(yo[j]); alerts=int(cw[j])-excess
                                oracle_tp,oracle_alerts,oracle_P=expanded_strict(w[oracle_order],oracle_y,K)
                                assert tp==oracle_tp and alerts==oracle_alerts==K and P==oracle_P
                                legacy_j=min(int(np.searchsorted(cw,(num/den)*n,side='left')),len(cw)-1)
                                old_tp=int(cp[legacy_j]);old_alerts=int(cw[legacy_j])
                                orig[ci_,ai,ri]=old_tp/P;fixed[ci_,ai,ri]=tp/P
                                checks['components_verified']+=1
                                checks['legacy_over_budget_components']+=old_alerts>K
                                checks['max_legacy_excess']=max(checks['max_legacy_excess'],old_alerts-K)
                                checks['legacy_recall_changed_components']+=old_tp!=tp
                                if ri==0 and ai>0:
                                    checks['rho_0.02_C_B_components']+=1
                                    checks['rho_0.02_C_B_over_budget']+=old_alerts>K
                    for pair,(a,c) in PAIRS.items():
                        for ri,rh in enumerate(RHOS):
                            legacy_draws[f,pair,rh][b,si]=np.mean(orig[:,a,ri]-orig[:,c,ri])
                            strict_draws[f,pair,rh][b,si]=np.mean(fixed[:,a,ri]-fixed[:,c,ri])
            if b%100==0:print(dataset,b,'elapsed',round(time.perf_counter()-started,1),flush=True)
        rows=[]
        for f in FAM:
            for pair,(a,c) in PAIRS.items():
                for rh in RHOS:
                    key='0.02' if rh=='rho_0.02' else rh
                    point_splits=np.array([point_cache[f,si,a,key]-point_cache[f,si,c,key] for si in range(5)])
                    old=archive['paired_recall_differences'][dataset][f][rh][pair]
                    original=legacy_draws[f,pair,rh];corrected=strict_draws[f,pair,rh]
                    original_ci=ci(original.mean(1));corrected_ci=ci(corrected.mean(1))
                    assert np.max(np.abs(np.array(original_ci)-old['ci95']))<1e-12
                    assert abs(point_splits.mean()-old['delta_mean'])<1e-12
                    for si in range(5):
                        assert np.max(np.abs(np.array(ci(original[:,si]))-old['per_split'][si]['ci95']))<1e-12
                    row={'dataset':dataset,'family':f,'pair':pair,'budget':rh,'point':float(point_splits.mean()),
                         'legacy_ci95':old['ci95'],'legacy_replay_ci95':original_ci,'strict_ci95':corrected_ci,
                         'endpoint_change':(np.array(corrected_ci)-np.array(old['ci95'])).tolist(),
                         'legacy_contains_zero':original_ci[0]<=0<=original_ci[1],
                         'strict_contains_zero':corrected_ci[0]<=0<=corrected_ci[1],
                         'changed_draws':int(np.count_nonzero(original.mean(1)!=corrected.mean(1))),
                         'per_split':[{'seed':SEEDS[si],'point':float(point_splits[si]),'legacy_ci95':ci(original[:,si]),'strict_ci95':ci(corrected[:,si])} for si in range(5)]}
                    rows.append(row);allrows.append(row)
        draw_path=OUT/f'{dataset}_paired_recall_draws.npz'
        np.savez_compressed(draw_path,**{f'legacy__{f}__{p}__{r}':x for (f,p,r),x in legacy_draws.items()},
                            **{f'strict__{f}__{p}__{r}':x for (f,p,r),x in strict_draws.items()})
        saved.append(draw_path)
        all_results[dataset]={'rows':rows,'verification':checks,'point_boundary_ties':ties,
                              'unique_rows':len(union),'unique_positives':len(pos),'weight_matrix_shape':list(weight_matrix.shape)}
    assert all(sha(Path(x['path']))==x['sha256'] for x in manifest)
    e5=[r for r in allrows if r['pair']=='C-B' and r['budget']=='rho_0.02']
    summary={'stage':'stage4_v1.0','status':'passed','new_training':False,'legacy_sources_unchanged':True,
             'protocol':{'n_bootstrap':B,'seed':20261101,'reset_seed_per_dataset':True,
                         'strict_rule':'K=ceil(rho*N_b), stable descending score then original stored row position, first K expanded occurrences; boundary copies truncated',
                         'budget_arithmetic':'integer ceiling; fixed rho=1/50 or fixed original split prevalence P/N',
                         'sampling':'unique-row class-stratified replacement; one shared row-weight vector per dataset replicate, across all arms, stress inputs and splits',
                         'aggregation':'recall difference per stress replica, average replicas within family then average five splits',
                         'independent_oracle':'lexsort(score,original_position), np.repeat each row label by its sampled multiplicity, first K occurrence labels; exact integer TP, alert count and denominator checked',
                         'inference_limit':'conditional on stored predictions, masks, selection; no training uncertainty or provenance certification'},
             'n_family_pair_intervals':len(allrows),'n_split_pair_intervals':5*len(allrows),
             'E5_max_abs_endpoint_change':max(abs(v) for r in e5 for v in r['endpoint_change']),
             'E5_zero_crossing_unchanged':all(r['legacy_contains_zero']==r['strict_contains_zero'] for r in e5),
             'all_interval_zero_crossing_unchanged':all(r['legacy_contains_zero']==r['strict_contains_zero'] for r in allrows),
             'E5_rows':e5,'datasets':all_results,'input_manifest':manifest,
             'output_manifest':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in saved],
             'runtime':{'executable':sys.executable,'numpy':np.__version__,'elapsed_seconds':time.perf_counter()-started}}
    (OUT/'strict_alert_bootstrap_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','n_family_pair_intervals','n_split_pair_intervals','E5_max_abs_endpoint_change','E5_zero_crossing_unchanged','all_interval_zero_crossing_unchanged']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
