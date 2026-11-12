"""独立校验导出件：从 CSV 重算 30 格 AP、检查 npz 键与掩码、核对行数。"""
import csv, json, os, sys, hashlib
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

RUN = "/home/user/workspace/默认项目/补A3_复现_2026-11-12/A3_repro_2026-11-12"
summary = json.load(open(os.path.join(RUN, "summary.json")))
rec = {}
for s in (11, 22, 33, 44, 55):
    for arm in ("C", "B"):
        for ms in (2026, 2027, 2028):
            p = f"/home/user/workspace/默认项目/新增实验_2026-11-05/results_seed/s{s}_{arm}_{ms}.json"
            rec[(s, arm, ms)] = json.load(open(p))["ap"]

groups = defaultdict(lambda: ([], []))
n = 0
with open(os.path.join(RUN, "a3_persample_predictions.csv"), encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        n += 1
        k = (int(row["split_seed"]), row["arm"], int(row["model_seed"]))
        groups[k][0].append(int(row["y_true"]))
        groups[k][1].append(float(row["prob"]))

print("CSV 行数:", n, "分组数:", len(groups))
bad = []
for k in sorted(groups):
    yy, pp = groups[k]
    ap = average_precision_score(yy, pp)
    if ap != rec[k]:
        bad.append((k, ap, rec[k]))
print("从 CSV 重算 AP 与历史记录不一致的格数:", len(bad))
for b in bad[:5]:
    print("  ", b)

# npz 结构
for s in (11, 22, 33, 44, 55):
    z = np.load(os.path.join(RUN, f"preds_s{s}.npz"))
    assert z["val_rows"].shape == (9000,), z["val_rows"].shape
    assert z["y_val"].shape == (9000,)
    for arm in ("C", "B"):
        for ms in (2026, 2027, 2028):
            a = z[f"{arm}_{ms}"]
            assert a.shape == (9000,), a.shape
    shape = tuple(z["delmask_shape"])
    for arm in ("C", "B"):
        packed = z[f"delmask_{arm}"]
        bits = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        print(f"s{s} {arm}: 新增遮蔽单元 {int(bits.sum())}, shape={shape}, packed={packed.shape}")

print("val_rows 唯一性:", all(len(set(np.load(os.path.join(RUN, f'preds_s{s}.npz'))['val_rows'].tolist())) == 9000 for s in (11, 22, 33, 44, 55)))
print("summary: cells=%d bit_equal=%d" % (summary["n_cells"], summary["n_bit_equal"]))
