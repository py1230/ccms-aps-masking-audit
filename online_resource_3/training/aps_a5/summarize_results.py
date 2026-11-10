"""Prespecified descriptive aggregation; no p values or inferential intervals."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent
GROUPS=['S0','S1_10','S1_30','S2','S3']
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    cfg=json.loads((HERE/'config.json').read_text());units=[]
    for split in cfg['split_seeds']:
        p=HERE/'run_results'/f'split{split}'/'summary.json';s=json.loads(p.read_text())
        assert s['status']=='complete' and s['fits']==15
        assert {r['repeat'] for r in s['rows']}==set(cfg['randomization_repeats'])
        for row in s['rows']:
            assert row['split']==split
            for group in GROUPS:
                units.append({'split':split,'repeat':row['repeat'],'group':group,'A5':row['ap'][group],
                    **{arm:row['reference_ap'][arm][group] for arm in ['C025','B025','BASE']},
                    **{arm:row['contrasts'][arm][group] for arm in ['C025-A5','A5-B025','A5-BASE']}})
    repeats=[];measures=['A5','C025','B025','BASE','C025-A5','A5-B025','A5-BASE']
    for rep in cfg['randomization_repeats']:
        for group in GROUPS:
            sub=[r for r in units if r['repeat']==rep and r['group']==group];assert len(sub)==5
            repeats.append({'repeat':rep,'group':group,**{m:float(np.mean([r[m] for r in sub])) for m in measures}})
    stats=[]
    for group in GROUPS:
        for measure in measures:
            values=[r[measure] for r in repeats if r['group']==group];assert len(values)==5
            stats.append({'group':group,'measure':measure,'mean':float(np.mean(values)),'min':float(np.min(values)),
                'max':float(np.max(values)),'repeat_values':values,'scope':'five finite-chain randomizations; descriptive range, not CI'})
    primary=next(r for r in stats if r['group']=='S1_30' and r['measure']=='C025-A5')
    csvwrite('ap_and_contrasts_by_unit.csv',units);csvwrite('ap_and_contrasts_by_repeat.csv',repeats)
    csvwrite('descriptive_summary.csv',[{k:v for k,v in r.items() if k not in ['repeat_values']} for r in stats])
    dump(HERE/'descriptive_summary.json',{'status':'complete','fits':75,'independent_randomization_repeats':5,
         'fixed_splits':5,'primary':primary,'results':stats,'inference':'no p values, CI, significance or equivalence tests'})
    print(json.dumps(primary),flush=True)
if __name__=='__main__':main()
