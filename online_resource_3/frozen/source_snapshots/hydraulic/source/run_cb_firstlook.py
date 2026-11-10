"""C/B 首轮探索性对照（协议 P1）。

只答一个问题：同 α、同骨干、同划分、同评价压力下，类别条件遮蔽 C 与类别盲预算
对照 B 的差异是否超出随机波动。

纪律（与 protocol/探索性协议_P1_CB首轮_2026-10-31.md 一致）：
- q 与填补器只在原始训练部分拟合，C/B/A0 共用；
- 遮蔽只加在训练矩阵；评价输入只做与标签无关的压力遮蔽；
- 不接触官方测试集；不据此选 α。

用法：
    python3 scripts/run_cb_firstlook.py --out records
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aps_io import dev_split, file_sha256, load_aps_csv, split_features  # noqa: E402
from audit_metrics import average_precision, auroc, evaluate_all  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer, _make_model  # noqa: E402

ALPHAS = (0.25, 0.5, 1.0)
MASK_SEEDS = (9107, 9108, 9109)
MODEL_SEED = 2026
N_BOOT = 500
BOOT_SEED = 20260731


# ---------------------------------------------------------------- 压力构造
def stress_unit(X, rate, seed):
    """S1：对可见单元额外遮蔽 rate。"""
    X = np.asarray(X, dtype=float)
    U = np.random.default_rng(seed).random(X.shape)
    new = (~np.isnan(X)) & (U < rate)
    Xn = X.copy()
    Xn[new] = np.nan
    return Xn, {"kind": "unit", "rate": float(rate), "seed": int(seed), "deleted": int(new.sum())}


def stress_rows(X, row_frac, cell_rate, seed):
    """S2：随机 row_frac 的行，其可见单元以 cell_rate 遮蔽。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rows = rng.choice(n, size=int(round(row_frac * n)), replace=False)
    U = rng.random((len(rows), X.shape[1]))
    Xn = X.copy()
    sub = Xn[rows]
    new = (~np.isnan(sub)) & (U < cell_rate)
    sub[new] = np.nan
    Xn[rows] = sub
    return Xn, {"kind": "rows", "row_frac": float(row_frac), "cell_rate": float(cell_rate),
                "seed": int(seed), "n_rows": int(len(rows)), "deleted": int(new.sum())}


def stress_cols(X, col_frac, seed):
    """S3：随机约 col_frac 的列整批不可用。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n_cols = X.shape[1]
    cols = rng.choice(n_cols, size=max(1, int(round(col_frac * n_cols))), replace=False)
    Xn = X.copy()
    before = np.isnan(Xn).sum()
    Xn[:, cols] = np.nan
    return Xn, {"kind": "cols", "col_frac": float(col_frac), "seed": int(seed),
                "n_cols_dropped": int(len(cols)), "deleted": int(np.isnan(Xn).sum() - before)}


def build_eval_inputs(X_val):
    """13 个评价输入，全部只依赖特征。返回 {name: (matrix, meta)}。"""
    out = {}
    out["S0"] = (np.asarray(X_val, dtype=float), {"kind": "original"})
    for s in (101, 202, 303):
        m, meta = stress_unit(X_val, 0.10, s)
        out[f"S1_10_{s}"] = (m, meta)
    for s in (101, 202, 303):
        m, meta = stress_unit(X_val, 0.30, s)
        out[f"S1_30_{s}"] = (m, meta)
    for s in (1101, 1202, 1303):
        m, meta = stress_rows(X_val, 0.50, 0.60, s)
        out[f"S2_{s}"] = (m, meta)
    for s in (2101, 2202, 2303):
        m, meta = stress_cols(X_val, 0.30, s)
        out[f"S3_{s}"] = (m, meta)
    return out


# ---------------------------------------------------------------- 训练与评价
def make_train_matrix(backbone, X_masked, imputer):
    if backbone == "M3":
        return np.hstack([imputer.transform(X_masked), rebuild_missing_flags(X_masked)])
    if backbone == "M4":
        return np.asarray(X_masked, dtype=float).copy()
    raise ValueError(backbone)


def predict(backbone, model, X_eval, imputer):
    return model.predict_proba(make_train_matrix(backbone, X_eval, imputer))[:, 1]


def topk_metrics(y, scores, k):
    y = np.asarray(y).ravel().astype(int)
    order = np.argsort(-np.asarray(scores).ravel(), kind="stable")[:k]
    tp = int(y[order].sum())
    return {"k": int(k), "tp": tp, "precision": tp / k, "recall": tp / max(1, int(y.sum()))}


def metrics_of(y, proba, k_list=(150, 300)):
    y = np.asarray(y).ravel().astype(int)
    rec = evaluate_all(y, proba, thresholds=(0.5,))
    rec["topk"] = {str(k): topk_metrics(y, proba, k) for k in k_list}
    rec.pop("at_threshold_note", None)
    return rec


def bootstrap_delta(y, p_a, p_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """按类别分层的配对 bootstrap：返回 ΔAP = AP(a) − AP(b) 的均值与 95% 百分位区间。"""
    y = np.asarray(y).ravel().astype(int)
    p_a = np.asarray(p_a, dtype=float).ravel()
    p_b = np.asarray(p_b, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    d = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(idx_pos, idx_pos.size, replace=True),
                              rng.choice(idx_neg, idx_neg.size, replace=True)])
        d[i] = average_precision(y[idx], p_a[idx]) - average_precision(y[idx], p_b[idx])
    point = average_precision(y, p_a) - average_precision(y, p_b)
    return {
        "delta": float(point),
        "boot_mean": float(d.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_le_0": float(np.mean(d <= 0.0)),
        "n_boot": int(n_boot),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "aps_failure_training_set.csv"))
    ap.add_argument("--out", default=str(ROOT / "records"))
    ap.add_argument("--alphas", default="0.25,0.5,1")
    ap.add_argument("--seeds", default="9107,9108,9109")
    args = ap.parse_args(argv)

    alphas = tuple(float(x) for x in args.alphas.split(","))
    mask_seeds = tuple(int(x) for x in args.seeds.split(","))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    df, meta = load_aps_csv(args.data)
    X, y = split_features(df)
    sp = dev_split(y, seed=7, stratify=True)
    X_tr, y_tr = X[sp["train"]], y[sp["train"]]
    X_va, y_va = X[sp["val"]], y[sp["val"]]

    imputer = MedianImputer().fit(X_tr)  # 只在原始训练矩阵上拟合
    chk_med = np.nanmedian(X_tr, axis=0)
    assert np.allclose(imputer.medians_, np.where(np.isfinite(chk_med), chk_med, 0.0)), \
        "填补器中位数与原始训练矩阵不一致"

    q0, counts0, miss0 = estimate_class_rates(X_tr, y_tr)
    eval_inputs = build_eval_inputs(X_va)

    runs = []
    probas = {}  # key -> {cond: proba}

    def do_train_eval(key, backbone, X_train_masked, note):
        t0 = time.perf_counter()
        M = make_train_matrix(backbone, X_train_masked, imputer)
        model = _make_model(MODEL_SEED)
        model.fit(M, y_tr)
        fit_s = time.perf_counter() - t0
        per_cond = {}
        t1 = time.perf_counter()
        for name, (mat, _meta) in eval_inputs.items():
            p = predict(backbone, model, mat, imputer)
            per_cond[name] = metrics_of(y_va, p)
            probas.setdefault(key, {})[name] = p
        pred_s = time.perf_counter() - t1
        runs.append({
            "key": key, "backbone": backbone, "note": note,
            "n_features_matrix": int(M.shape[1]),
            "train_missing_rate_after_mask": float(np.isnan(X_train_masked).mean()),
            "fit_seconds": float(fit_s), "eval_seconds": float(pred_s),
            "metrics": per_cond,
        })
        return per_cond

    # 锚点
    do_train_eval("A0_M3", "M3", X_tr, {"alpha": 0.0, "method": "anchor"})
    do_train_eval("A0_M4", "M4", X_tr, {"alpha": 0.0, "method": "anchor"})

    deletions = []
    for backbone in ("M3", "M4"):
        for alpha in alphas:
            b_vec = class_blind_rates(q0, counts0, alpha)
            for sd in mask_seeds:
                Xc, info_c = mask_class_conditional(X_tr, y_tr, alpha, seed=sd)
                key_c = f"C{int(round(alpha * 100)):03d}_M{backbone[-1]}_s{sd}"
                do_train_eval(key_c, backbone, Xc, {"alpha": alpha, "method": "C", "mask_seed": sd})
                Xb, info_b = mask_class_blind(X_tr, b_vec, seed=sd)
                key_b = f"B{int(round(alpha * 100)):03d}_M{backbone[-1]}_s{sd}"
                do_train_eval(key_b, backbone, Xb, {"alpha": alpha, "method": "B", "mask_seed": sd})
                deletions.append({
                    "backbone": backbone, "alpha": alpha, "mask_seed": sd,
                    "C_deleted_total": int(info_c["deleted_total"]),
                    "B_deleted_total": int(info_b["deleted_total"]),
                    "C_deleted_per_class": [int(v) for v in info_c["deleted_per_class_col"].sum(axis=1)],
                    "B_deleted_total_expected": float(np.sum(b_vec * (~np.isnan(X_tr)).sum(axis=0))),
                    "n_visible": int((~np.isnan(X_tr)).sum()),
                })

    # 三个遮蔽重复的概率平均（ens3）
    def ens(key_base):
        ks = [f"{key_base}_s{sd}" for sd in mask_seeds]
        return {cond: np.mean([probas[k][cond] for k in ks], axis=0) for cond in eval_inputs}

    comparisons = []
    for backbone in ("M3", "M4"):
        for alpha in alphas:
            tag = f"{int(round(alpha * 100)):03d}"
            p_c = ens(f"C{tag}_M{backbone[-1]}")
            p_b = ens(f"B{tag}_M{backbone[-1]}")
            p_0 = probas[f"A0_{backbone}"]
            for cond in eval_inputs:
                row = {"backbone": backbone, "alpha": alpha, "condition": cond}
                row["AP_C"] = float(average_precision(y_va, p_c[cond]))
                row["AP_B"] = float(average_precision(y_va, p_b[cond]))
                row["AP_A0"] = float(average_precision(y_va, p_0[cond]))
                row["AUROC_C"] = float(auroc(y_va, p_c[cond]))
                row["AUROC_B"] = float(auroc(y_va, p_b[cond]))
                row["AUROC_A0"] = float(auroc(y_va, p_0[cond]))
                row["dAP_C_minus_B"] = row["AP_C"] - row["AP_B"]
                row["dAP_C_minus_A0"] = row["AP_C"] - row["AP_A0"]
                row["dAP_B_minus_A0"] = row["AP_B"] - row["AP_A0"]
                if alpha == 0.5 or backbone == "M3":
                    row["boot_C_minus_B"] = bootstrap_delta(y_va, p_c[cond], p_b[cond])
                comparisons.append(row)

    def runs_ap(key, cond):
        for r in runs:
            if r["key"] == key:
                return float(r["metrics"][cond]["ap"])
        raise KeyError(key)

    # 遮蔽随机性带来的方差：逐重复 AP 的极差
    seed_spread = []
    for backbone in ("M3", "M4"):
        for alpha in alphas:
            tag = f"{int(round(alpha * 100)):03d}"
            for cond in eval_inputs:
                aps_c = [runs_ap(f"C{tag}_M{backbone[-1]}_s{sd}", cond)
                         for sd in mask_seeds]
                aps_b = [runs_ap(f"B{tag}_M{backbone[-1]}_s{sd}", cond)
                         for sd in mask_seeds]
                seed_spread.append({
                    "backbone": backbone, "alpha": alpha, "condition": cond,
                    "C_ap_per_seed": aps_c, "C_ap_range": max(aps_c) - min(aps_c),
                    "B_ap_per_seed": aps_b, "B_ap_range": max(aps_b) - min(aps_b),
                })

    record = {
        "protocol": "探索性协议_P1_CB首轮_2026-10-31.md",
        "run_tier": "exploratory_dev_only",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": float(time.time() - t_start),
        "data": meta,
        "data_sha256": file_sha256(args.data),
        "split": {"sizes": sp["sizes"], "positives": sp["positives"], "seed": 7, "stratified": True},
        "q_train_shape": list(q0.shape),
        "train_missing_rate": float(np.isnan(X_tr).mean()),
        "alphas": list(alphas),
        "mask_seeds": list(mask_seeds),
        "model_seed": MODEL_SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "eval_inputs": {k: v[1] for k, v in eval_inputs.items()},
        "deletions": deletions,
        "runs": runs,
        "comparisons": comparisons,
        "seed_spread": seed_spread,
        "notes": [
            "所有 AP 都是开发验证集（9000 行 / 150 正类）上的结果，不是正式测试结果。",
            "bootstrap 条件于已训练模型；不覆盖重新拟合与换划分的不确定性。",
            "ens3 = 三个遮蔽重复的预测概率算术平均。",
        ],
    }

    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"cb_firstlook_{stamp}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"written: {path}")

    # 摘要表：S0 与两个主压力条件
    print("\ncondition  backbone  alpha   AP_C     AP_B     AP_A0    C-B      C-A0")
    for row in comparisons:
        if row["condition"] in ("S0", "S1_30_101", "S2_1101", "S3_2101"):
            print(f"{row['condition']:<10} {row['backbone']:<8} {row['alpha']:<6} "
                  f"{row['AP_C']:.6f} {row['AP_B']:.6f} {row['AP_A0']:.6f} "
                  f"{row['dAP_C_minus_B']:+.6f} {row['dAP_C_minus_A0']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
