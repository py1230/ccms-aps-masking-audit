"""Frozen, descriptive bootstrap aggregation; anchor excluded from 20 repeats."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent
GROUPS=['S0','S1_10','S1_30','S2','S3']
def read(p):return json.loads(p.read_text())
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    cfg=read(HERE/'config.json');names=['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']]
    allrows=[read(HERE/'units'/name/'summary.json') for name in names]
    assert all(r['status']=='complete' and r['fits']==19 and r['unit']==name for r,name in zip(allrows,names))
    anchor=allrows[0];repeats=allrows[1:];assert len(repeats)==20
    aps=[];contrasts=[];choices=[]
    for r in allrows:
        for arm,values in r['ap'].items():
            for group,ap in values.items():aps.append({'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'config':arm,'group':group,'AP':ap})
        for contrast,values in r['contrasts'].items():
            for group,difference in values.items():contrasts.append({'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'contrast':contrast,'group':group,'difference':difference})
        choices.append({'unit':r['unit'],'bootstrap_seed':r['bootstrap_seed'],'C_alpha':r['chosen_alphas']['C'],'B_alpha':r['chosen_alphas']['B']})
    stats=[]
    for kind in ['ap','contrasts']:
        for measure in anchor[kind]:
            for group in GROUPS:
                values=[r[kind][measure][group] for r in repeats];mean=float(np.mean(values));baseline=anchor[kind][measure][group]
                stats.append({'kind':kind,'measure':measure,'group':group,'n':20,'mean':mean,'sample_sd':float(np.std(values,ddof=1)),
                    'min':float(np.min(values)),'max':float(np.max(values)),'anchor':baseline,'mean_minus_anchor':mean-baseline,'values':values})
    counts={arm:{str(alpha):sum(r['chosen_alphas'][arm]==alpha for r in repeats) for alpha in cfg['alphas']} for arm in ['C','B']}
    primary=next(r for r in stats if r['kind']=='contrasts' and r['measure']=='C_selected-B_selected' and r['group']=='S1_30')
    secondary=next(r for r in stats if r['kind']=='contrasts' and r['measure']=='C025-B025' and r['group']=='S1_30')
    dump(HERE/'descriptive_summary.json',{'status':'complete','formal_fits':sum(r['fits'] for r in allrows),'anchor_fits':19,'bootstrap_fits':380,
        'bootstrap_repeats':20,'primary':primary,'key_secondary':secondary,'selection_counts':counts,'anchor_choices':anchor['chosen_alphas'],
        'results':stats,'scope':'conditional empirical stratified training resampling at fixed validation/test; ranges are not confidence intervals; no p values'})
    table('ap_by_unit.csv',aps);table('contrasts_by_unit.csv',contrasts);table('selections.csv',choices)
    table('descriptive_summary.csv',[{k:v for k,v in r.items() if k!='values'} for r in stats])
    print(json.dumps({'primary':primary,'key_secondary':secondary,'selection_counts':counts}),flush=True)
if __name__=='__main__':main()
