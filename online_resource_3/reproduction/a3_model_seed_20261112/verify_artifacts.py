"""Portable checks of A3 reproduction files; no training or deserialization."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
root=Path(__file__).resolve().parent;resource=root.parents[1]
received=root/'received';run=received/'02_复现件_模型与掩码_非原件/A3_repro_2026-11-12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
count=0
for line in (received/'04_SHA256SUMS.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    expected,rel=line.split('  ',1);p=received/rel
    assert p.resolve().is_relative_to(received.resolve())
    assert sha(p)==expected,rel
    count+=1
groups={}
with (run/'a3_persample_predictions.csv').open(newline='') as f:
    for r in csv.DictReader(f):groups.setdefault((int(r['split_seed']),r['arm'],int(r['model_seed'])),[]).append(r)
cells={(c['split'],c['arm'],c['model_seed']):c for c in json.loads((run/'summary.json').read_text())['cells']}
assert set(groups)==set(cells)=={(s,a,m) for s in [11,22,33,44,55] for a in ['B','C'] for m in [2026,2027,2028]}
diffs=[];masks=[]
for s in [11,22,33,44,55]:
    with np.load(run/f'preds_s{s}.npz',allow_pickle=False) as z,np.load(resource/f'data/aps_predictions/aps_seed{s}.npz',allow_pickle=False) as old:
        assert np.array_equal(z['val_rows'],old['val_rows']) and np.array_equal(z['y_val'],old['y_val'])
        shape=tuple(map(int,z['delmask_shape']));assert shape==(42000,170)
        for ai,a in enumerate(['C','B']):
            bits=np.unpackbits(z[f'delmask_{a}'])[:np.prod(shape)].reshape(shape).astype(bool)
            mh=hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest();assert mh==str(z['mask_sha256'][ai])
            masks.append(int(bits.sum()))
            for m in [2026,2027,2028]:
                k=(s,a,m);rows=sorted(groups[k],key=lambda r:int(r['val_index']))
                assert [int(r['val_index']) for r in rows]==list(range(9000))
                p=np.array([float(r['prob']) for r in rows]);y=np.array([int(r['y_true']) for r in rows]);idx=np.array([int(r['source_row_0based']) for r in rows])
                assert np.array_equal(p,z[f'{a}_{m}']) and np.array_equal(y,z['y_val']) and np.array_equal(idx,z['val_rows'])
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                order=np.argsort(-p,kind='stable');last=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1];tp=np.cumsum(y[order])[last]
                ap=float(np.sum(np.diff(np.r_[0,tp])*tp/(last+1))/sum(y))
                rec=json.loads((received/f'01_当次执行记录_原件/results_seed/s{s}_{a}_{m}.json').read_text())
                diff=abs(ap-rec['ap']);assert diff<1e-12;diffs.append(diff)
                c=cells[k];assert c['mask_sha256']==mh and sha(run/c['model_file'])==c['model_sha256']
print(json.dumps({'status':'PASS','classification':'2026-11-12 reproduction','manifest_entries':count,'prediction_groups':len(diffs),'prediction_rows':sum(map(len,groups.values())),'training_masks':len(masks),'max_abs_ap_difference':max(diffs),'models_checked_by_hash_only':True,'model_inference_replayed':False,'deterministic_masks_regenerated_by_this_portable_check':False}))
