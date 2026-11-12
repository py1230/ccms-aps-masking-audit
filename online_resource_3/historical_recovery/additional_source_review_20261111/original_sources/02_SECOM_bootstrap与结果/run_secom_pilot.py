"""SECOM pilot：外层 5 折、固定 α、出折预测池化评价。

目的：在第二份真实数据上测（1）基线可用性与真实耗时内存；（2）C 与 B 在固定 α 下的方向
是否与 APS 一致；（3）缺失标志单独能排多前。**不是正式结果**：α 未选、只 5 折一次、单遮蔽种子 3 次。

用法：python3 scripts/run_secom_pilot.py
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
ALPHAS = (0.25, 0.5)
MASK_SEEDS = (9107, 9108, 9109)


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


def fit_eval(backbone, imp, Xtr, ytr, conds):
    M = matrix_for(backbone, imp, Xtr)
    model = _make_model(2026) if backbone != "M5L" else None
    t0 = time.perf_counter()
    if backbone == "M5L":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(M, ytr)
    fit_s = time.perf_counter() - t0
    out = {}
    for name, Xev in conds.items():
        p = model.predict_proba(matrix_for(backbone, imp, Xev))[:, 1]
        out[name] = p
    return out, fit_s, M.shape[1]


def main():
    t_start = time.time()
    X, y, meta = load_secom(DATA, LABELS)
    prof = profile(X, y)
    folds = stratified_kfold(y, k=5, seed=11)

    configs = ["A0_M3", "A0_M4", "M5L", "M5T"]
    for a in ALPHAS:
        for meth in ("C", "B"):
            configs.append(f"{meth}{int(round(a*100)):03d}_M3")

    oof = {c: {} for c in configs}
    per_fold = []
    total_fit = 0.0

    for fi, (tr, te) in enumerate(folds):
        keep = select_columns_in_train(X[tr])
        Xtr_all, Xte_all = X[tr][:, keep], X[te][:, keep]
        ytr, yte = y[tr], y[te]
        imp = MedianImputer().fit(Xtr_all)
        conds = {"S0": Xte_all, "S1_10": extra_unit(Xte_all, 0.10, 1000 + fi)}
        fold_rec = {"fold": fi, "n_train": int(len(tr)), "n_test": int(len(te)),
                    "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
                    "n_cols_kept": int(len(keep)), "runs": []}

        def store(cfg, backbone, Xtr_used):
            probs, fit_s, nfeat = fit_eval(backbone, imp, Xtr_used, ytr, conds)
            for name, p in probs.items():
                arr = oof[cfg].setdefault(name, np.full(len(y), np.nan))
                arr[te] = p
                fold_rec["runs"].append({
                    "config": cfg, "condition": name, "n_features": int(nfeat),
                    "fit_seconds": float(fit_s),
                    "ap": float(average_precision(yte, p)), "auroc": float(auroc(yte, p)),
                })

        store("A0_M3", "M3", Xtr_all)
        store("A0_M4", "M4", Xtr_all)
        store("M5L", "M5L", Xtr_all)
        store("M5T", "FLAGS", Xtr_all)

        q0, counts0, _ = estimate_class_rates(Xtr_all, ytr)
        for a in ALPHAS:
            bvec = class_blind_rates(q0, counts0, a)
            for ms in MASK_SEEDS:
                Xc, _ = mask_class_conditional(Xtr_all, ytr, a, seed=ms)
                store(f"C{int(round(a*100)):03d}_M3", "M3", Xc)
                Xb, _ = mask_class_blind(Xtr_all, bvec, seed=ms)
                store(f"B{int(round(a*100)):03d}_M3", "M3", Xb)

        total_fit += sum(r["fit_seconds"] for r in fold_rec["runs"])
        per_fold.append(fold_rec)
        print(f"fold {fi}: train {len(tr)} ({int(ytr.sum())}+) test {len(te)} ({int(yte.sum())}+) "
              f"cols {len(keep)} done")

    pooled = {}
    for cfg, conds in oof.items():
        pooled[cfg] = {}
        for name, arr in conds.items():
            ok = ~np.isnan(arr)
            pooled[cfg][name] = {
                "ap": float(average_precision(y[ok], arr[ok])),
                "auroc": float(auroc(y[ok], arr[ok])),
                "n": int(ok.sum()), "n_pos": int(y[ok].sum()),
            }

    rec = {
        "run_tier": "pilot_5fold_fixed_alpha",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": float(time.time() - t_start),
        "total_fit_seconds": float(total_fit),
        "meta": meta, "profile": prof,
        "folds": [{"fold": f["fold"], "n_train": f["n_train"], "n_test": f["n_test"],
                   "pos_train": f["pos_train"], "pos_test": f["pos_test"],
                   "n_cols_kept": f["n_cols_kept"]} for f in per_fold],
        "per_fold": per_fold,
        "pooled": pooled,
        "notes": [
            "固定 α，不选 α；每折内不重训全量；出折预测池化后算 AP/AUROC。",
            "常量列与全空列只在当前训练折判断；掩码不参与评价。",
            "正类共 104 个，单折只有约 21 个，逐折 AP 波动极大，结论看池化值。",
        ],
    }
    path = ROOT / "records" / f"secom_pilot_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n池化（{pooled['A0_M3']['S0']['n']} 行 / {pooled['A0_M3']['S0']['n_pos']} 正类）")
    print(f"{'config':<10}{'AP(S0)':>10}{'AUROC(S0)':>12}{'AP(S1_10)':>12}{'AUROC(S1)':>12}")
    for cfg in configs:
        print(f"{cfg:<10}{pooled[cfg]['S0']['ap']:>10.4f}{pooled[cfg]['S0']['auroc']:>12.4f}"
              f"{pooled[cfg]['S1_10']['ap']:>12.4f}{pooled[cfg]['S1_10']['auroc']:>12.4f}")
    print(f"\n总拟合 {total_fit:.1f}s，总墙钟 {rec['wall_seconds']:.0f}s")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
