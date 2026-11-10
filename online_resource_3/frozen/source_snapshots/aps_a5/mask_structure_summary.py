"""Post hoc descriptive mask structure diagnostics; no performance access."""
from pathlib import Path
import csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from aps_io import load_aps_csv,split_features
def main():
    config=json.loads((HERE/'config.json').read_text());d,_=load_aps_csv(HERE/'raw_data/aps_failure_training_set.csv');x,y=split_features(d)
    out=HERE/'structure_diagnostics';out.mkdir(exist_ok=False);rows=[];per_reference=[]
    for split in config['split_seeds']:
        root=HERE/'reference'/f'split{split}';tr=np.load(root/'preprocessing.npz')['train_rows'];labels=y[tr];visible=~np.isnan(x[tr]);masks=[]
        for rs in config['reference_mask_seeds']:
            ref=np.load(root/'training_masks'/f'C025_seed{rs}.npz')['new_mask'];refclass=np.stack([ref[labels==c].sum(0) for c in [0,1]])
            refroot=HERE/'feasibility'/f'split{split}'/f'reference{rs}'
            div=json.loads((refroot/'diversity.json').read_text());per_reference.append({'split':split,'reference_seed':rs,'pairwise_min':min(div['pairwise_hamming']),'pairwise_max':max(div['pairwise_hamming']),'distinct_masks':div['distinct_masks'],'legal_space_lower_bound':div['legal_space_lower_bound']})
            for rep in config['randomization_repeats']:
                folder=refroot/f'rep{rep}';mask=np.load(folder/'mask.npz')['new_mask'];meta=json.loads((folder/'chain.json').read_text())
                newclass=np.stack([mask[labels==c].sum(0) for c in [0,1]]);diff=newclass.astype(int)-refclass.astype(int)
                assert np.array_equal(newclass.sum(1),refclass.sum(1))
                changed=mask!=ref;co=np.array(meta['diagnostics'][-1]['cooccurrence_upper']);co0=np.array(meta['diagnostics'][0]['cooccurrence_upper']);coprev=np.array(meta['diagnostics'][-2]['cooccurrence_upper'])
                row={'split':split,'reference_seed':rs,'repeat':rep,'changed_rows':int(changed.any(1).sum()),'changed_row_fraction':float(changed.any(1).mean()),
                     'changed_cols':int(changed.any(0).sum()),'changed_col_fraction':float(changed.any(0).mean()),
                     'class_col_l1_change':int(np.abs(diff).sum()),'class_col_max_change':int(np.abs(diff).max()),
                     'class_col_nonzero_changes':int(np.count_nonzero(diff)),'cooccurrence16_l1_from_C':int(np.abs(co-co0).sum()),
                     'cooccurrence16_l1_10E_to_20E':int(np.abs(co-coprev).sum()),'cooccurrence16_changed_pairs_10E_to_20E':int(np.count_nonzero(co-coprev)),
                     'relocated_fraction':meta['diagnostics'][-1]['relocated_fraction']}
                for c in [0,1]:
                    row[f'class{c}_total_deleted_C']=int(refclass[c].sum());row[f'class{c}_total_deleted_A5']=int(newclass[c].sum())
                    row[f'class{c}_visible_cells']=int(visible[labels==c].sum());row[f'class{c}_deleted_per_visible']=float(newclass[c].sum()/visible[labels==c].sum())
                rows.append(row)
                np.savez_compressed(out/f'split{split}_reference{rs}_rep{rep}.npz',class_col_deleted_C=refclass,class_col_deleted_A5=newclass,
                    class_col_visible=np.stack([visible[labels==c].sum(0) for c in [0,1]]),row_deleted_C=ref.sum(1),row_deleted_A5=mask.sum(1),row_class=labels)
    measures=[k for k in rows[0] if k not in ['split','reference_seed','repeat']]
    ranges={k:{'min':min(r[k] for r in rows),'max':max(r[k] for r in rows),'mean':float(np.mean([r[k] for r in rows]))} for k in measures}
    result={'status':'complete','scope':'exploratory mask-only diagnostics, no performance; no ESS/Rhat/mixing inference','chains':len(rows),
       'class_total_deleted_exactly_preserved':True,'row_counts_and_their_class_association':'exactly preserved by fixed per-row deletion counts; per-class total/visible-cell rate remains identical to C',
       'joint_structure_scope':'class by column plus first 16 active columns pair cooccurrence; not all joint structures',
       'ranges':ranges,'pairwise_hamming_min':min(r['pairwise_min'] for r in per_reference),'pairwise_hamming_max':max(r['pairwise_max'] for r in per_reference),
       'rows':rows,'per_reference':per_reference}
    (HERE/'mask_structure_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'mask_structure_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'chains':len(rows),'class_total_deleted_exactly_preserved':True,'ranges':ranges}),flush=True)
if __name__=='__main__':main()
