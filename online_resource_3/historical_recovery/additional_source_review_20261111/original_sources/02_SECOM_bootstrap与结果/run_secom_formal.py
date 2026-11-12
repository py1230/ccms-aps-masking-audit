"""SECOM 正式五折（protocol v2 口径）。

与 pilot 的差别：
- α 网格 {0.25, 0.5, 1.0}（pilot 只有 0.25/0.5），每个 α 跑 3 个遮蔽种子；
- 三个评价条件 S0 / S1_10 / S1_30（pilot 只有前两个）；
- 相减区间：在折内按行重采样后池化（fold-stratified bootstrap），比较 C−B、C−A0、B−A0；
- 记录每折的列选择、样本数、正类数、拟合耗时，便于复现。

用法：python3 scripts/run_secom_formal.py [--boot 1000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_metrics import average_precision, auroc  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer, _make_model  # noqa: E402
from secom_io import load_secom, profile, select_columns_in_train, stratified_kfold  # noqa: E402

DATA = str(ROOT.parent / "data" / "secom" / "secom.data")
LABELS = str(ROOT.parent / "data" / "secom" / "secom_labels.data")
ALPHAS = (0.25, 0.5, 1.0)
MASK_SEEDS = (9107, 9108, 9109)
CONDS = ("S0", "S1_10", "S1_30")


def extra_unit(X, rate, seed):
    U = np.random.default_rng(seed).random(X.shape)
    new = (~np.isnan(X)) & (U < rate)
    Xn = X.copy()
    Xn[new] = np.nan
    return Xn


def matrix_for(backbone, imp, X):
    if backbone == "M3":
        return np.hstack([imp.transform(X), rebuild_missing_flags(X)])
    if backbone == "M4":
        return np.asarray(X, dtype=float).copy()
    if backbone in ("FLAGS", "M5L", "M5T"):
        return rebuild_missing_flags(X)
    raise ValueError(backbone)


def pooled_ap(y, p, idx):
    return float(average_precision(y[idx], p[idx]))


def fold_stratified_bootstrap(y, p_a, p_b, fold_of, n_boot=1000, seed=4242):
    """折内按行重采样（保持折大小与折内正负比例随机性），返回 ΔAP 的区间。"""
    rng = np.random.default_rng(seed)
    folds = [np.flatnonzero(fold_of == k) for k in np.unique(fold_of)]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(f, size=len(f), replace=True) for f in folds])
        deltas[b] = pooled_ap(y, p_a, idx) - pooled_ap(y, p_b, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(pooled_ap(y, p_a, np.arange(len(y))) -
                           pooled_ap(y, p_b, np.arange(len(y)))),
            "ci95": [float(lo), float(hi)],
            "n_boot": int(n_boot), "resample": "fold_stratified_rows"}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)

    t_start = time.time()
    X, y, meta = load_secom(DATA, LABELS)
    prof = profile(X, y)
    folds = stratified_kfold(y, k=5, seed=11)

    configs = ["A0_M3", "A0_M4", "M5L", "M5T"]
    for a in ALPHAS:
        for meth in ("C", "B"):
            configs.append(f"{meth}{int(round(a * 100)):03d}_M3")

    oof = {c: {k: np.full(len(y), np.nan) for k in CONDS} for c in configs}
    fold_of = np.full(len(y), -1)
    per_fold, total_fit = [], 0.0

    for fi, (tr, te) in enumerate(folds):
        fold_of[te] = fi
        keep = select_columns_in_train(X[tr])
        Xtr, Xte = X[tr][:, keep], X[te][:, keep]
        ytr, yte = y[tr], y[te]
        imp = MedianImputer().fit(Xtr)
        conds = {"S0": Xte,
                 "S1_10": extra_unit(Xte, 0.10, 1000 + fi),
                 "S1_30": extra_unit(Xte, 0.30, 2000 + fi)}
        rec = {"fold": fi, "n_train": int(len(tr)), "n_test": int(len(te)),
               "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
               "n_cols_kept": int(len(keep)), "runs": []}

        def store(cfg, backbone, Xtr_used):
            M = matrix_for(backbone, imp, Xtr_used)
            model = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
                     if backbone == "M5L" else _make_model(2026))
            t = time.perf_counter()
            model.fit(M, ytr)
            fit_s = time.perf_counter() - t
            row = {"config": cfg, "n_features": int(M.shape[1]), "fit_seconds": float(fit_s)}
            for name, Xev in conds.items():
                p = model.predict_proba(matrix_for(backbone, imp, Xev))[:, 1]
                oof[cfg][name][te] = p
                row[f"ap_{name}"] = float(average_precision(yte, p))
                row[f"auroc_{name}"] = float(auroc(yte, p))
            rec["runs"].append(row)
            return fit_s

        for cfg, bb in (("A0_M3", "M3"), ("A0_M4", "M4"), ("M5L", "M5L"), ("M5T", "FLAGS")):
            total_fit += store(cfg, bb, Xtr)

        q0, counts0, _ = estimate_class_rates(Xtr, ytr)
        for a in ALPHAS:
            bvec = class_blind_rates(q0, counts0, a)
            tag = int(round(a * 100))
            for ms in MASK_SEEDS:
                Xc, _ = mask_class_conditional(Xtr, ytr, a, seed=ms)
                total_fit += store(f"C{tag:03d}_M3", "M3", Xc)
                Xb, _ = mask_class_blind(Xtr, bvec, seed=ms)
                total_fit += store(f"B{tag:03d}_M3", "M3", Xb)

        per_fold.append(rec)
        print(f"fold {fi}: train {len(tr)} ({int(ytr.sum())}+) test {len(te)} ({int(yte.sum())}+) "
              f"cols {len(keep)} done ({time.time()-t_start:.0f}s)", flush=True)

    pooled = {c: {k: {"ap": pooled_ap(y, oof[c][k], np.arange(len(y))),
                      "auroc": float(auroc(y, oof[c][k]))} for k in CONDS} for c in configs}

    # 逐折 AP 的离散程度（正类少，必须看）
    fold_ap = {c: {k: [r for r in [0]] for k in CONDS} for c in configs}
    for c in configs:
        for k in CONDS:
            vals = []
            for rec in per_fold:
                for run in rec["runs"]:
                    if run["config"] == c:
                        vals.append(run[f"ap_{k}"])
            fold_ap[c][k] = vals

    comparisons = {}
    for a in ALPHAS:
        tag = int(round(a * 100))
        for cond in CONDS:
            key = f"alpha{a}"
            comparisons.setdefault(key, {})
            comparisons[key][cond] = {
                "C-B": fold_stratified_bootstrap(y, oof[f"C{tag:03d}_M3"][cond],
                                                 oof[f"B{tag:03d}_M3"][cond], fold_of, args.boot),
                "C-A0": fold_stratified_bootstrap(y, oof[f"C{tag:03d}_M3"][cond],
                                                  oof["A0_M3"][cond], fold_of, args.boot),
                "B-A0": fold_stratified_bootstrap(y, oof[f"B{tag:03d}_M3"][cond],
                                                  oof["A0_M3"][cond], fold_of, args.boot),
            }

    rec = {
        "run_tier": "formal_5fold_v2",
        "protocol": "protocol_v2_冻结_2026-10-31.md",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": float(time.time() - t_start),
        "total_fit_seconds": float(total_fit),
        "meta": meta, "profile": prof,
        "alphas": list(ALPHAS), "mask_seeds": list(MASK_SEEDS), "conditions": list(CONDS),
        "n_bootstrap": int(args.boot),
        "folds": [{"fold": r["fold"], "n_train": r["n_train"], "n_test": r["n_test"],
                   "pos_train": r["pos_train"], "pos_test": r["pos_test"],
                   "n_cols_kept": r["n_cols_kept"]} for r in per_fold],
        "per_fold": per_fold,
        "pooled": pooled,
        "fold_ap_spread": fold_ap,
        "comparisons": comparisons,
        "notes": [
            "外层 5 折（种子 11），列选择、填补、q 估计都只在当前训练折内拟合。",
            "每个 α 的 C/B 各跑 3 个遮蔽种子，出折预测先按种子平均再入池。",
            "C vs B 的遮蔽预算按类别盲规则匹配（b_j 用训练折的类别比例加权）。",
            "正类 104 个、单折约 21 个，逐折 AP 波动大；主看池化值与折内重采样区间。",
        ],
    }
    path = ROOT / "records" / f"secom_formal_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n池化（{len(y)} 行 / {int(y.sum())} 正类）")
    print(f"{'config':<12}{'AP(S0)':>9}{'AUROC(S0)':>11}{'AP(S1_10)':>11}{'AP(S1_30)':>11}")
    for c in configs:
        p = pooled[c]
        print(f"{c:<12}{p['S0']['ap']:>9.4f}{p['S0']['auroc']:>11.4f}"
              f"{p['S1_10']['ap']:>11.4f}{p['S1_30']['ap']:>11.4f}")
    for key, block in comparisons.items():
        for cond, comp in block.items():
            s = f"{key} {cond}: " + "  ".join(
                f"{lab} {v['delta']:+.4f}[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                for lab, v in comp.items())
            print(s)
    print(f"\n总拟合 {total_fit:.0f}s，墙钟 {rec['wall_seconds']:.0f}s\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
