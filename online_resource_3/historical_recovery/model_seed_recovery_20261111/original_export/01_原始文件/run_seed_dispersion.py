"""模型种子离散度：M3 + 默认 HGB，S1 30%，C/B 各三个模型种子。"""
import json, os, sys, time
import numpy as np
SRC="/home/user/workspace/默认项目/CCMS核查_2026-10-31/src"; SCR="/home/user/workspace/默认项目/CCMS核查_2026-10-31/scripts"
sys.path.insert(0,SRC); sys.path.insert(0,SCR)
from aps_io import load_aps_csv, split_features, dev_split
from ccms import estimate_class_rates, class_blind_rates, mask_class_conditional, mask_class_blind, rebuild_missing_flags
from pipeline import MedianImputer, _make_model
from run_cb_firstlook import build_eval_inputs
from sklearn.metrics import average_precision_score

OUT="/home/user/workspace/默认项目/新增实验_2026-11-05/results_seed"
os.makedirs(OUT, exist_ok=True)
df,_=load_aps_csv("/home/user/workspace/默认项目/aps_failure_training_set.csv")
X,y=split_features(df)
for s in (11,22,33,44,55):
    sp=dev_split(y,seed=s,stratify=True)
    Xtr,ytr=X[sp["train"]],y[sp["train"]]
    Xva,yva=X[sp["val"]],y[sp["val"]]
    imp=MedianImputer().fit(Xtr); ev=build_eval_inputs(Xva)["S1_30_101"][0]
    q,cnt,_=estimate_class_rates(Xtr,ytr); b=class_blind_rates(q,cnt,0.25)
    for arm,Xm in (("C",mask_class_conditional(Xtr,ytr,0.25,seed=9107)[0]),("B",mask_class_blind(Xtr,b,seed=9107)[0])):
        M=np.hstack([imp.transform(Xm), rebuild_missing_flags(Xm)])
        Mv=np.hstack([imp.transform(ev), rebuild_missing_flags(ev)])
        for ms in (2026,2027,2028):
            p=f"{OUT}/s{s}_{arm}_{ms}.json"
            if os.path.exists(p): continue
            m=_make_model(ms); m.fit(M,ytr)
            a=float(average_precision_score(yva, m.predict_proba(Mv)[:,1]))
            json.dump({"split":s,"arm":arm,"model_seed":ms,"ap":a}, open(p,"w"))
            print(s,arm,ms,round(a,4), flush=True)
print("SEED_DONE")
