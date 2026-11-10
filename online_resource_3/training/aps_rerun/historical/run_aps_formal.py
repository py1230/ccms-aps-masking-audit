"""APS 正式运行（protocol v2 冻结版）：五划分 × 13 配置 × 13 评价输入。

按 protocol/protocol_v2_冻结_2026-10-31.md 执行：
- 划分种子 11/22/33/44/55，分层；每划分重新拟合预处理与 q；
- α 选择只用 S0 与三个 10% 单元缺失副本；
- 配对 bootstrap 1000 次，主终结点为 S1_30 与 S2。

不含官方测试集（数据不在本机），那一步等数据到位后按同一脚本的 --test 分支补齐。

用法：python3 scripts/run_aps_formal.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aps_io import dev_split, load_aps_csv, split_features  # noqa: E402
from audit_metrics import average_precision, auroc  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer, _make_model  # noqa: E402
from run_cb_firstlook import build_eval_inputs, bootstrap_delta, metrics_of  # noqa: E402

DATA = str(ROOT.parent / "aps_failure_training_set.csv")
SPLIT_SEEDS = (11, 22, 33, 44, 55)
ALPHAS = (0.25, 0.5, 1.0)
MASK_SEEDS = (9107, 9108, 9109)
SEL_CONDS = ("S0", "S1_10_101", "S1_10_202", "S1_10_303")
BOOT_CONDS = ("S0", "S1_30_101", "S2_1101", "S3_2101")


def build(backbone, imp, X):
    if backbone == "M3":
        return np.hstack([imp.transform(X), rebuild_missing_flags(X)])
    if backbone == "M2":
        return imp.transform(X)
    if backbone == "M4":
        return np.asarray(X, dtype=float).copy()
    if backbone == "FLAGS":
        return rebuild_missing_flags(X)
    raise ValueError(backbone)


def make_model(cfg):
    if cfg == "M0":
        return DummyClassifier(strategy="prior")
    if cfg == "M1":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if cfg == "M5L":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return _make_model(2026)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,22,33,44,55")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    split_seeds = tuple(int(s) for s in args.seeds.split(","))
    t0 = time.time()
    df, meta = load_aps_csv(DATA)
    X, y = split_features(df)
    out = {"protocol": "protocol_v2_冻结_2026-10-31.md",
           "run_tier": "formal_5splits",
           "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "data_sha256": meta["sha256"],
           "split_seeds": list(split_seeds), "alphas": list(ALPHAS),
           "n_bootstrap": int(args.boot),
           "mask_seeds": list(MASK_SEEDS), "splits": []}

    for seed in split_seeds:
        ts = time.time()
        sp = dev_split(y, seed=seed, stratify=True)
        X_tr, y_tr = X[sp["train"]], y[sp["train"]]
        X_va, y_va = X[sp["val"]], y[sp["val"]]
        imp = MedianImputer().fit(X_tr)
        evals = build_eval_inputs(X_va)

        proba = {}
        metrics = {}
        fit_info = {}

        def run_cfg(cfg, backbone, Xtr_used, seeds=(None,)):
            ps = {name: [] for name in evals}
            for sd in seeds:
                M = build(backbone, imp, Xtr_used)
                model = make_model(cfg)
                t = time.perf_counter()
                model.fit(M, y_tr)
                fit_s = time.perf_counter() - t
                for name, (mat, _m) in evals.items():
                    ps[name].append(model.predict_proba(build(backbone, imp, mat))[:, 1])
                fit_info[cfg] = {"fit_seconds": float(fit_s), "n_features": int(M.shape[1]),
                                 "n_train_rows": int(M.shape[0])}
            proba[cfg] = {k: np.mean(v, axis=0) for k, v in ps.items()}
            metrics[cfg] = {k: metrics_of(y_va, proba[cfg][k]) for k in evals}

        run_cfg("A0_M3", "M3", X_tr)
        metrics["M3"] = metrics.pop("A0_M3")
        proba["M3"] = proba.pop("A0_M3")
        fit_info["M3"] = fit_info.pop("A0_M3")

        for cfg, backbone in (("M0", "M2"), ("M1", "M1"), ("M2", "M2"), ("M4", "M4"),
                              ("M5L", "FLAGS"), ("M5T", "FLAGS")):
            if cfg == "M1":
                M = imp.transform(X_tr)
                model = make_model("M1")
                model.fit(M, y_tr)
                ps = {name: model.predict_proba(imp.transform(mat))[:, 1]
                      for name, (mat, _m) in evals.items()}
                proba["M1"] = ps
                metrics["M1"] = {k: metrics_of(y_va, ps[k]) for k in evals}
                fit_info["M1"] = {"n_features": int(M.shape[1]), "n_train_rows": int(M.shape[0])}
            else:
                run_cfg(cfg, backbone, X_tr)

        q0, counts0, _ = estimate_class_rates(X_tr, y_tr)
        for alpha in ALPHAS:
            bvec = class_blind_rates(q0, counts0, alpha)
            tag = f"{int(round(alpha * 100)):03d}"
            ps_c, ps_b = {k: [] for k in evals}, {k: [] for k in evals}
            fit_c, fit_b = [], []
            for sd in MASK_SEEDS:
                Xc, _ = mask_class_conditional(X_tr, y_tr, alpha, seed=sd)
                Mc = build("M3", imp, Xc)
                mc = _make_model(2026)
                t = time.perf_counter()
                mc.fit(Mc, y_tr)
                fit_c.append(time.perf_counter() - t)
                Xb, _ = mask_class_blind(X_tr, bvec, seed=sd)
                Mb = build("M3", imp, Xb)
                mb = _make_model(2026)
                t = time.perf_counter()
                mb.fit(Mb, y_tr)
                fit_b.append(time.perf_counter() - t)
                for name, (mat, _m) in evals.items():
                    M = build("M3", imp, mat)
                    ps_c[name].append(mc.predict_proba(M)[:, 1])
                    ps_b[name].append(mb.predict_proba(M)[:, 1])
            for tag2, ps, fs in ((f"C{tag}", ps_c, fit_c), (f"B{tag}", ps_b, fit_b)):
                proba[tag2] = {k: np.mean(v, axis=0) for k, v in ps.items()}
                metrics[tag2] = {k: metrics_of(y_va, proba[tag2][k]) for k in evals}
                fit_info[tag2] = {"fit_seconds_mean": float(np.mean(fs)),
                                  "n_features": int(Mc.shape[1]),
                                  "n_train_rows": int(Mc.shape[0])}

        # α 选择
        def select(prefix):
            base_ap = metrics["M3"]["S0"]["ap"]
            cands = []
            for alpha in ALPHAS:
                tag = f"{int(round(alpha * 100)):03d}"
                cfg = f"{prefix}{tag}"
                ok = metrics[cfg]["S0"]["ap"] >= base_ap - 0.01
                worst = min(metrics[cfg][c]["ap"] for c in SEL_CONDS)
                cands.append({"alpha": alpha, "cfg": cfg, "s0_ap": metrics[cfg]["S0"]["ap"],
                              "worst4": worst, "eligible": bool(ok)})
            elig = [c for c in cands if c["eligible"]]
            if not elig:
                return {"chosen_alpha": 0.0, "chosen_cfg": "M3", "candidates": cands}
            best = max(elig, key=lambda c: (c["worst4"], -c["alpha"]))
            return {"chosen_alpha": best["alpha"], "chosen_cfg": best["cfg"],
                    "candidates": cands}

        sel_c, sel_b = select("C"), select("B")
        ci = {}
        for cname in BOOT_CONDS:
            ci[cname] = {}
            for label, a, b in (("C-B", sel_c["chosen_cfg"], sel_b["chosen_cfg"]),
                                ("C-A0", sel_c["chosen_cfg"], "M3"),
                                ("B-A0", sel_b["chosen_cfg"], "M3")):
                ci[cname][label] = bootstrap_delta(y_va, proba[a][cname], proba[b][cname],
                                                   n_boot=args.boot)

        out["splits"].append({
            "seed": seed, "sizes": sp["sizes"], "positives": sp["positives"],
            "selection": {"C": sel_c, "B": sel_b},
            "fit_info": fit_info,
            "metrics": metrics,
            "bootstrap": ci,
            "elapsed_seconds": float(time.time() - ts),
        })
        print(f"split {seed}: C*={sel_c['chosen_alpha']} B*={sel_b['chosen_alpha']} "
              f"S0 AP M3={metrics['M3']['S0']['ap']:.4f} "
              f"C*={metrics[sel_c['chosen_cfg']]['S0']['ap']:.4f} "
              f"B*={metrics[sel_b['chosen_cfg']]['S0']['ap']:.4f} ({time.time()-ts:.0f}s)")

    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["wall_seconds"] = float(time.time() - t0)
    path = ROOT / "records" / f"aps_formal_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
