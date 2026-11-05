"""Robustness check with alternative model families (supplementary analysis of protocol v2; does not change the main results).

Question: do the main results (C below B, B no worse than the backbone) hold only for HistGradientBoosting?
Design: replace the backbone with ExtraTrees and LightGBM and keep everything else fixed (same splits, same masking seeds,
same alpha, same evaluation inputs, same M3-style matrix = median imputation plus all-column missingness indicators).

Usage: python3 scripts/run_robustness.py [--boot 1000] [--folds 5]
Output: records/robustness_<ts>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aps_io import dev_split, load_aps_csv, split_features  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer  # noqa: E402
from run_cb_firstlook import bootstrap_delta, build_eval_inputs, metrics_of  # noqa: E402

DATA = str(ROOT.parent / "aps_failure_training_set.csv")
ALPHAS = (0.25, 1.0)
MASK_SEEDS = (9107, 9108)
SPLIT_SEEDS = (11, 22, 33)
EVAL_CONDS = ("S0", "S1_30_101", "S2_1101", "S3_2101")


def make_model(family, seed=2026):
    if family == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=150, max_features="sqrt",
                                    min_samples_leaf=3, n_jobs=2, random_state=seed)
    if family == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  subsample=0.9, colsample_bytree=0.8, n_jobs=2,
                                  random_state=seed, verbose=-1)
    raise ValueError(family)


def build(imp, X):
    return np.hstack([imp.transform(X), rebuild_missing_flags(X)])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args(argv)

    t0 = time.time()
    df, meta = load_aps_csv(DATA)
    X, y = split_features(df)
    out = {"run_tier": "robustness_model_family",
           "purpose": "cross-check the relative direction of C and B under another model family; changes neither the main results nor the protocol",
           "data_sha256": meta["sha256"],
           "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "split_seeds": list(SPLIT_SEEDS)[:args.folds],
           "alphas": list(ALPHAS), "mask_seeds": list(MASK_SEEDS),
           "eval_conditions": list(EVAL_CONDS), "n_bootstrap": int(args.boot),
           "families": {}, "splits": []}

    for family in ("ExtraTrees", "LightGBM"):
        out["families"][family] = {"params": ("n_estimators=150, max_features=sqrt, min_samples_leaf=3, n_jobs=2"
                                             if family == "ExtraTrees" else
                                             "n_estimators=400, lr=0.05, num_leaves=31, "
                                             "subsample=0.9, colsample=0.8, n_jobs=2"),
                                   "random_state": 2026}

    for seed in list(SPLIT_SEEDS)[:args.folds]:
        ts = time.time()
        sp = dev_split(y, seed=seed, stratify=True)
        X_tr, y_tr = X[sp["train"]], y[sp["train"]]
        X_va, y_va = X[sp["val"]], y[sp["val"]]
        imp = MedianImputer().fit(X_tr)
        evals_all = build_eval_inputs(X_va)
        evals = {k: evals_all[k] for k in EVAL_CONDS}
        q0, counts0, _ = estimate_class_rates(X_tr, y_tr)
        rec = {"seed": seed, "sizes": sp["sizes"], "positives": sp["positives"], "families": {}}

        for family in ("ExtraTrees", "LightGBM"):
            proba, metrics, info = {}, {}, {}
            m = make_model(family)
            t = time.perf_counter()
            m.fit(build(imp, X_tr), y_tr)
            info["A0"] = {"fit_seconds": round(time.perf_counter() - t, 2)}
            proba["A0"] = {k: m.predict_proba(build(imp, v[0]))[:, 1] for k, v in evals.items()}
            metrics["A0"] = {k: metrics_of(y_va, proba["A0"][k]) for k in evals}

            for alpha in ALPHAS:
                tag = f"{int(round(alpha * 100)):03d}"
                bvec = class_blind_rates(q0, counts0, alpha)
                for arm in ("C", "B"):
                    ps = {k: [] for k in evals}
                    fs = []
                    for sd in MASK_SEEDS:
                        if arm == "C":
                            Xm, _ = mask_class_conditional(X_tr, y_tr, alpha, seed=sd)
                        else:
                            Xm, _ = mask_class_blind(X_tr, bvec, seed=sd)
                        mdl = make_model(family)
                        t = time.perf_counter()
                        mdl.fit(build(imp, Xm), y_tr)
                        fs.append(time.perf_counter() - t)
                        for k, (mat, _m) in evals.items():
                            ps[k].append(mdl.predict_proba(build(imp, mat))[:, 1])
                    cfg = f"{arm}{tag}"
                    proba[cfg] = {k: np.mean(v, axis=0) for k, v in ps.items()}
                    metrics[cfg] = {k: metrics_of(y_va, proba[cfg][k]) for k in evals}
                    info[cfg] = {"fit_seconds_mean": round(float(np.mean(fs)), 2)}

            ci = {}
            for cond in EVAL_CONDS:
                ci[cond] = {}
                for label, a, b in (("C-B", "C025", "B025"),
                                    ("C-A0", "C025", "A0"),
                                    ("B-A0", "B025", "A0"),
                                    ("C1.0-A0", "C100", "A0")):
                    ci[cond][label] = bootstrap_delta(y_va, proba[a][cond], proba[b][cond],
                                                      n_boot=args.boot)
            rec["families"][family] = {"fit_info": info, "metrics": metrics, "bootstrap": ci}
            print(f"split {seed} {family}: S0 A0={metrics['A0']['S0']['ap']:.4f} "
                  f"C025={metrics['C025']['S0']['ap']:.4f} B025={metrics['B025']['S0']['ap']:.4f} "
                  f"C100={metrics['C100']['S0']['ap']:.4f} | "
                  f"S1_30 A0={metrics['A0']['S1_30_101']['ap']:.4f} "
                  f"C025={metrics['C025']['S1_30_101']['ap']:.4f} "
                  f"B025={metrics['B025']['S1_30_101']['ap']:.4f}", flush=True)

        rec["elapsed_seconds"] = round(time.time() - ts, 1)
        out["splits"].append(rec)
        # write after every fold so that an interrupted long run cannot lose all evidence
        part = ROOT / "records" / f"robustness_partial_{time.strftime('%Y%m%dT%H%M%S')}.json"
        part.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  checkpoint -> {part.name}", flush=True)

    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["wall_seconds"] = round(time.time() - t0, 1)
    path = ROOT / "records" / f"robustness_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {path}\nwall {out['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
