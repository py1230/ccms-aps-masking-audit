"""Check A3 reproduction predictions against archived labels and APs."""
from pathlib import Path
import csv,io,json,hashlib,zipfile,argparse
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--esm-zip',type=Path);p.add_argument('--csv',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent;resource=root.parents[1]
z=zipfile.ZipFile(a.esm_zip) if a.esm_zip else None
csv_path=a.csv or root/'a3_model_seed_predictions.csv'
expected='61f2290707ac98b09a9d2ea55e66040eda30a0b08e8b03f0ad37ed12473383e0'
assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==expected
records={}
with csv_path.open(newline='') as f:
 for r in csv.DictReader(f):records.setdefault((int(r['split_seed']),r['arm'],int(r['model_seed'])),[]).append(r)
assert set(records)=={(s,arm,m) for s in [11,22,33,44,55] for arm in ['B','C'] for m in [2026,2027,2028]}
def src(name):return z.read('online_resource_3/'+name) if z else (resource/name).read_bytes()
def ap(y,prob):
 order=np.argsort(-prob,kind='stable');score=prob[order];y=y[order]
 last=np.r_[np.flatnonzero(np.diff(score)),len(y)-1];tp=np.cumsum(y)[last]
 return float(np.sum(np.diff(np.r_[0,tp])*tp/(last+1))/sum(y))
out=[]
for (s,arm,m),rows in sorted(records.items()):
 rows.sort(key=lambda r:int(r['val_index']))
 saved=np.load(io.BytesIO(src(f'data/aps_predictions/aps_seed{s}.npz')),allow_pickle=False)
 y=np.array([int(r['y_true']) for r in rows]);prob=np.array([float(r['prob']) for r in rows]);idx=np.array([int(r['source_row_0based']) for r in rows])
 assert len(rows)==9000 and [int(r['val_index']) for r in rows]==list(range(9000))
 assert np.array_equal(idx,saved['val_rows']) and np.array_equal(y,saved['y_val'])
 assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
 h=json.loads(src(f'historical_recovery/model_seed_recovery_20261111/original_export/01_原始文件/results_seed/s{s}_{arm}_{m}.json'))
 value=ap(y,prob);diff=abs(value-h['ap']);assert diff<1e-12
 out.append({'split':s,'arm':arm,'model_seed':m,'ap':value,'historical_ap':h['ap'],'abs_difference':diff})
r={'all_pass':True,'sha256':expected,'rows':sum(len(x) for x in records.values()),'groups':out,'max_abs_ap_difference':max(x['abs_difference'] for x in out),'provenance':'2026-11-12 reproduction output; earlier original identity corrected by author. Numerical checks do not authenticate historical execution.'}
if a.out:a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps({'all_pass':True,'groups':len(out),'max_abs_ap_difference':r['max_abs_ap_difference']}))
