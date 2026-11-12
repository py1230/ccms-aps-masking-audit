"""A3 模型种子实验：同配置再执行 + 逐样本导出（独立 run ID）。

身份说明：这是复现件，不是 2026-11-04 当次运行落盘的原件。
参数与 run_seed_dispersion.py 完全一致：划分 11/22/33/44/55、alpha=0.25、
C/B 掩码种子 9107、评价输入 S1_30_101（划分内验证集）、模型种子 2026/2027/2028。
"""
import json, os, sys, time, hashlib, platform
import numpy as np

RUN_ID = "A3_repro_2026-11-12"
ROOT = "/home/user/workspace/默认项目"
OUT = os.path.join(ROOT, "补A3_复现_2026-11-12", RUN_ID)
SRC = os.path.join(ROOT, "CCMS核查_2026-10-31/src")
SCR = os.path.join(ROOT, "CCMS核查_2026-10-31/scripts")
sys.path.insert(0, SRC); sys.path.insert(0, SCR)

from aps_io import load_aps_csv, split_features, dev_split
from ccms import estimate_class_rates, class_blind_rates, mask_class_conditional, mask_class_blind, rebuild_missing_flags
from pipeline import MedianImputer, _make_model
from run_cb_firstlook import build_eval_inputs
from sklearn.metrics import average_precision_score
import sklearn

os.makedirs(OUT, exist_ok=True)
STATUS = os.path.join(OUT, "status.json")
LOG = os.path.join(OUT, "run.log")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def sha(b):
    return hashlib.sha256(b).hexdigest()

t0 = time.time()
log(f"RUN_ID={RUN_ID} start")
env = {
    "run_id": RUN_ID,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "script": os.path.abspath(__file__),
    "script_sha256": sha(open(__file__, "rb").read()),
    "ccms_py_sha256": sha(open(os.path.join(SRC, "ccms.py"), "rb").read()),
    "pipeline_py_sha256": sha(open(os.path.join(SRC, "pipeline.py"), "rb").read()),
    "aps_csv_sha256": sha(open(os.path.join(ROOT, "aps_failure_training_set.csv"), "rb").read()),
    "params": {"splits": [11, 22, 33, 44, 55], "alpha": 0.25, "mask_seed": 9107,
               "model_seeds": [2026, 2027, 2028], "eval_input": "S1_30_101",
               "arm_C": "mask_class_conditional", "arm_B": "mask_class_blind"},
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(env, open(os.path.join(OUT, "environment.json"), "w"), indent=2)
log("environment.json written")

df, _ = load_aps_csv(os.path.join(ROOT, "aps_failure_training_set.csv"))
X, y = split_features(df)
log(f"data loaded X={X.shape} pos={int(y.sum())}")

records = {}
for s in (11, 22, 33, 44, 55):
    p = os.path.join(ROOT, "新增实验_2026-11-05/results_seed", f"s{s}_C_2026.json")
    for arm in ("C", "B"):
        for ms in (2026, 2027, 2028):
            fp = os.path.join(ROOT, "新增实验_2026-11-05/results_seed", f"s{s}_{arm}_{ms}.json")
            records[(s, arm, ms)] = json.load(open(fp))["ap"]

rows = []   # 逐样本长表
cells = []
all_match = True

for s in (11, 22, 33, 44, 55):
    sp = dev_split(y, seed=s, stratify=True)
    tr, va = sp["train"], sp["val"]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    imp = MedianImputer().fit(Xtr)
    ev = build_eval_inputs(Xva)["S1_30_101"][0]
    Mv = np.hstack([imp.transform(ev), rebuild_missing_flags(ev)])
    q, cnt, _ = estimate_class_rates(Xtr, ytr)
    b = class_blind_rates(q, cnt, 0.25)

    preds = {}
    del_masks = {}
    mask_hashes = {}
    for arm, Xm in (("C", mask_class_conditional(Xtr, ytr, 0.25, seed=9107)[0]),
                    ("B", mask_class_blind(Xtr, b, seed=9107)[0])):
        del_mask = np.isnan(Xm) & ~np.isnan(Xtr)          # 本次新增遮蔽的单元格
        del_masks[arm] = np.packbits(del_mask, axis=None)
        mask_hashes[arm] = sha(np.ascontiguousarray(del_mask).tobytes())
        M = np.hstack([imp.transform(Xm), rebuild_missing_flags(Xm)])
        for ms in (2026, 2027, 2028):
            t1 = time.time()
            m = _make_model(ms); m.fit(M, ytr)
            prob = m.predict_proba(Mv)[:, 1]
            ap = float(average_precision_score(yva, prob))
            rec = records[(s, arm, ms)]
            match = (ap == rec)
            all_match &= match
            preds[f"{arm}_{ms}"] = prob
            # 保存模型对象
            import joblib
            mf = os.path.join(OUT, f"model_s{s}_{arm}_{ms}.joblib")
            joblib.dump(m, mf)
            cells.append({"split": s, "arm": arm, "model_seed": ms, "ap": ap,
                          "recorded_ap": rec, "bit_equal": match,
                          "abs_diff": abs(ap - rec), "seconds": round(time.time() - t1, 2),
                          "mask_sha256": mask_hashes[arm],
                          "model_file": os.path.basename(mf),
                          "model_sha256": sha(open(mf, "rb").read())})
            log(f"s{s} {arm} {ms} ap={ap!r} rec={rec!r} match={match}")
            for i, r in enumerate(va):
                rows.append((s, arm, ms, i, int(r), int(yva[i]), float(prob[i])))

    np.savez_compressed(
        os.path.join(OUT, f"preds_s{s}.npz"),
        val_rows=va, y_val=yva,
        **{k: v for k, v in preds.items()},
        **{f"delmask_{a}": del_masks[a] for a in del_masks},
        delmask_shape=np.array(Xtr.shape),
        mask_sha256=np.array([mask_hashes["C"], mask_hashes["B"]]),
    )
    log(f"split {s} npz written; masks C={mask_hashes['C'][:12]} B={mask_hashes['B'][:12]}")
    json.dump({"split": s, "cells": [c for c in cells if c["split"] == s]},
              open(os.path.join(OUT, f"cells_s{s}.json"), "w"), indent=2)

# 逐样本 CSV
import csv
csvp = os.path.join(OUT, "a3_persample_predictions.csv")
with open(csvp, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split_seed", "arm", "model_seed", "val_index", "source_row_0based", "y_true", "prob"])
    w.writerows(rows)
log(f"csv written rows={len(rows)}")

summary = {
    "run_id": RUN_ID,
    "identity": "reproduction artifact, not the original 2026-11-04 output",
    "cells": cells,
    "n_cells": len(cells),
    "n_bit_equal": sum(1 for c in cells if c["bit_equal"]),
    "all_bit_equal": bool(all_match),
    "csv_rows": len(rows),
    "elapsed_seconds": round(time.time() - t0, 1),
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
json.dump({"state": "done", "n_bit_equal": summary["n_bit_equal"],
           "elapsed_seconds": summary["elapsed_seconds"]}, open(STATUS, "w"), indent=2)
log(f"ALL_DONE cells={len(cells)} bit_equal={summary['n_bit_equal']} elapsed={summary['elapsed_seconds']}s")
