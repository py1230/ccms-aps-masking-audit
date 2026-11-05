"""Supplementary experiments requested by the review report: native-NaN backbone, early-stopping sensitivity, missingness-only baseline, ExtraTrees.

Protocol follows the CCMS archive: five splits 11/22/33/44/55 (stratified by dev_split), alpha = 0.25,
three masking seeds 9107/9108/9109, model seed 2026. The archived code is read only and never modified.
One record per (split, arm, seed); the run can be resumed after an interruption.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

SRC = "/home/user/workspace/默认项目/CCMS核查_2026-10-31/src"
SCR = "/home/user/workspace/默认项目/CCMS核查_2026-10-31/scripts"
sys.path.insert(0, SRC)
sys.path.insert(0, SCR)

from aps_io import load_aps_csv, split_features, dev_split  # noqa: E402
from ccms import (class_blind_rates, estimate_class_rates,  # noqa: E402
                  mask_class_blind, mask_class_conditional, rebuild_missing_flags)
from pipeline import MedianImputer, _make_model  # noqa: E402
from run_cb_firstlook import build_eval_inputs, bootstrap_delta  # noqa: E402

DATA = "/home/user/workspace/默认项目/aps_failure_training_set.csv"
OUTDIR = "/home/user/workspace/默认项目/新增实验_2026-11-05/results"
ALPHA = 0.25
MASK_SEEDS = (9107, 9108, 9109)
HEAD = ("S0", "S1_30_101", "S2_1101", "S3_2101")


def ap(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y).ravel(), np.asarray(p).ravel()))


def model_for(kind, early_stopping):
    if kind == "ET":
        return ExtraTreesClassifier(n_estimators=100, random_state=2026, n_jobs=2)
    if early_stopping:
        return _make_model(2026)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=2026, early_stopping=False, max_iter=100)


def to_input(backbone, imp, X):
    X = np.asarray(X, dtype=float)
    if backbone == "M4":
        return X
    if backbone == "M3":
        return np.hstack([imp.transform(X), rebuild_missing_flags(X)])
    if backbone == "FLAGS":
        return rebuild_missing_flags(X)
    raise ValueError(backbone)


def arm_matrices(arm, Xtr, ytr, seed):
    """Return the masked training matrix (before conversion to model inputs) and the deletion statistics of that arm."""
    if arm == "none":
        return Xtr.copy(), {"deleted_total": 0, "arm": arm}
    q, counts, _ = estimate_class_rates(Xtr, ytr)
    if arm == "C":
        Xm, info = mask_class_conditional(Xtr, ytr, ALPHA, seed=seed)
    else:
        b = class_blind_rates(q, counts, ALPHA)
        Xm, info = mask_class_blind(Xtr, b, seed=seed)
    return Xm, {"deleted_total": int(info["deleted_total"]), "arm": arm, "seed": seed}


def run_split(exp, seed_split, X, y):
    sp = dev_split(y, seed=seed_split, stratify=True)
    Xtr, ytr = X[sp["train"]], y[sp["train"]]
    Xva, yva = X[sp["val"]], y[sp["val"]]
    imp = MedianImputer().fit(Xtr)
    evals = build_eval_inputs(Xva)

    if exp == "A":
        backbone, kind, es, arms = "M4", "HGB", True, ("none", "C", "B")
    elif exp == "B":
        backbone, kind, es, arms = "M3", "HGB", False, ("none", "C", "B")
    elif exp == "C":
        backbone, kind, es, arms = "FLAGS", "HGB", True, ("none", "C", "B")
    elif exp == "D":
        backbone, kind, es, arms = "M3", "ET", True, ("none", "C", "B")
    else:
        raise ValueError(exp)

    out = []
    for arm in arms:
        seeds = (None,) if arm == "none" else MASK_SEEDS
        for ms in seeds:
            rec_path = os.path.join(OUTDIR, f"{exp}_s{seed_split}_{arm}_{ms}.json")
            if os.path.exists(rec_path):
                out.append(json.load(open(rec_path)))
                continue
            t0 = time.time()
            Xm, dinfo = arm_matrices(arm, Xtr, ytr, ms)
            M = to_input(backbone, imp, Xm)
            model = model_for(kind, es)
            model.fit(M, ytr)
            fit_s = time.time() - t0
            aps = {}
            for name in evals:
                p = model.predict_proba(to_input(backbone, imp, evals[name][0]))[:, 1]
                aps[name] = ap(yva, p)
            rec = {"exp": exp, "backbone": backbone, "model": kind,
                   "early_stopping": bool(es), "split": seed_split, "arm": arm,
                   "mask_seed": ms, "alpha": ALPHA, "fit_s": round(fit_s, 1),
                   "deleted_total": dinfo["deleted_total"], "ap": aps}
            json.dump(rec, open(rec_path, "w"), indent=1)
            out.append(rec)
            print(f"{exp} split{seed_split} {arm} seed{ms} {fit_s:.0f}s "
                  f"S0={aps['S0']:.4f} S1_30={aps['S1_30_101']:.4f}", flush=True)
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--exp", required=True, choices=list("ABCD"))
    ap_.add_argument("--splits", default="11,22,33,44,55")
    args = ap_.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    df, meta = load_aps_csv(DATA)
    X, y = split_features(df)
    print(f"loaded {X.shape} pos={int(y.sum())} sha={meta['sha256'][:12]}", flush=True)
    for s in [int(v) for v in args.splits.split(",")]:
        run_split(args.exp, s, X, y)
    print(f"done {args.exp} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
