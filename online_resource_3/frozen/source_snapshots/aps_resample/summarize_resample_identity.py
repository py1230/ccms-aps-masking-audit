"""Descriptive bootstrap identity diagnostics; no performance access."""
from pathlib import Path
import csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from aps_io import load_aps_csv,split_features
def main():
    cfg=json.loads((HERE/'config.json').read_text());d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');_,y=split_features(d);rows=[]
    for name in ['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']]:
        z=np.load(HERE/'prepared_indices'/f'{name}.npz');ids=z['train_rows'];mult=z['multiplicity'];u=np.flatnonzero(mult)
        r={'unit':name,'draws':len(ids),'unique_original_rows':len(u),'duplicate_occurrences':len(ids)-len(u),'overall_max_multiplicity':int(mult.max()),'val_overlap_original_IDs':int(len(np.intersect1d(u,z['val_rows']))),'reserve_overlap_original_IDs':int(len(np.intersect1d(u,z['reserve_rows'])))}
        for c in [0,1]:
            v=mult[y==c];r[f'class{c}_draws']=int(v.sum());r[f'class{c}_unique']=int(np.count_nonzero(v));r[f'class{c}_max_multiplicity']=int(v.max());r[f'class{c}_IDs_with_copies_gt1']=int(np.count_nonzero(v>1))
        rows.append(r)
    reps=rows[1:];measures=[k for k in rows[0] if k!='unit'];ranges={k:{'min':min(r[k] for r in reps),'max':max(r[k] for r in reps),'mean':float(np.mean([r[k] for r in reps]))} for k in measures}
    out={'scope':'performance-blind identity diagnostics only; repeated copies are not new independent observations','bootstrap_repeats':20,'anchor_separate':True,'ranges':ranges,'anchor':rows[0],'rows':rows}
    (HERE/'resample_identity_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'resample_identity_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(ranges),flush=True)
if __name__=='__main__':main()
