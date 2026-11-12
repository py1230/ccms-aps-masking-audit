"""单格复现校验：s11 / C / model_seed=2026 / alpha=0.25 / mask_seed=9107 / S1_30_101。
仅用于诊断，输出写在 _tmp_a3_check/，不改动任何既有记录。"""
import sys, json
import numpy as np
SRC = "/home/user/workspace/默认项目/CCMS核查_2026-10-31/src"
SCR = "/home/user/workspace/默认项目/CCMS核查_2026-10-31/scripts"
sys.path.insert(0, SRC); sys.path.insert(0, SCR)
from aps_io import load_aps_csv, split_features, dev_split
from ccms import estimate_class_rates, mask_class_conditional, rebuild_missing_flags
from pipeline import MedianImputer, _make_model
from run_cb_firstlook import build_eval_inputs
from sklearn.metrics import average_precision_score

df, _ = load_aps_csv("/home/user/workspace/默认项目/aps_failure_training_set.csv")
X, y = split_features(df)
sp = dev_split(y, seed=11, stratify=True)
Xtr, ytr = X[sp["train"]], y[sp["train"]]
Xva, yva = X[sp["val"]], y[sp["val"]]
imp = MedianImputer().fit(Xtr)
ev = build_eval_inputs(Xva)["S1_30_101"][0]
q, cnt, _ = estimate_class_rates(Xtr, ytr)
Xm, info = mask_class_conditional(Xtr, ytr, 0.25, seed=9107)
M = np.hstack([imp.transform(Xm), rebuild_missing_flags(Xm)])
Mv = np.hstack([imp.transform(ev), rebuild_missing_flags(ev)])
m = _make_model(2026); m.fit(M, ytr)
ap = float(average_precision_score(yva, m.predict_proba(Mv)[:, 1]))
recorded = 0.461184996450304
a_arr = np.ravel(np.asarray(info["a"], dtype=float))
out = {
    "ap": ap, "recorded": recorded, "abs_diff": abs(ap - recorded),
    "bit_equal": ap == recorded,
    "n_train": int(len(ytr)), "n_val": int(len(yva)),
    "pos_train": int(ytr.sum()), "pos_val": int(yva.sum()),
    "mask_deleted_cells": int(np.isnan(Xm).sum() - np.isnan(Xtr).sum()),
    "mask_seed": info.get("seed"), "alpha_first": float(a_arr[0]),
    "sklearn": __import__("sklearn").__version__,
}
json.dump(out, open("/home/user/workspace/默认项目/_tmp_a3_check/result.json", "w"), indent=2)
print(json.dumps(out, indent=2))
