import summarize as s
import numpy as np
from sklearn.metrics import average_precision_score as AP
packs=[np.load(s.R/f'localization_seed{v}.npz') for v in [11,22,33,44,55]];out={}
for fam,conds in {'S0':['S0'],'S1_30':['S1_30_101','S1_30_202','S1_30_303']}.items():
 vals={c:[float(np.mean([AP(z['y'],z[c+'__'+k]) for k in conds])) for z in packs] for c in ['C_local','B_local']}
 out[fam]={'C_mean':float(np.mean(vals['C_local'])),'B_mean':float(np.mean(vals['B_local'])),'per_split_delta':(np.array(vals['C_local'])-vals['B_local']).tolist()}
s.save('localization_summary.json',out);print(out)
