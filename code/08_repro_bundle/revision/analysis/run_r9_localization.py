#!/usr/bin/env python
"""R9 supplement / B3: conditional intervals for the local localization ablation (per-column deletion counts matched exactly).

Input: results/localization_seed{11,22,33,44,55}.npz
  - per-row out-of-fold scores of y, validation_rows, C_local and B_local on S0 and the three S1_30 replicates.
  - intervention: keep only the positive-class deletions of the eight columns with the highest expected positive-class training deletions;
    the class-blind control deletes exactly the same number of cells uniformly among the visible cells of each column (counts equal per column).
  - the model probabilities of the three training-masking seeds (9107/9108/9109) are already averaged within a split (ens3),
    so the resampling here covers validation-row sampling only, not masking-seed or retraining uncertainty.

Output: revision/analysis/r9_localization_intervals.json
No model is refitted.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2] / "CCMS论文修订完整资料包_2026-11-01_192149"
RES = PKG / "02_本轮代码与结果/results"
OUT_DIR = Path(__file__).resolve().parents[1] / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
COPIES = ["S1_30_101", "S1_30_202", "S1_30_303"]
N_BOOT = 1000
BOOT_SEED = 20261101


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rnd(x, n=6):
    return float(np.round(float(x), n))


def prep(y: np.ndarray, p: np.ndarray):
    """Sort by score and merge tied values for weighted AP (same preparation as summarize.py)."""
    o = np.argsort(-p, kind="stable")
    last = np.r_[np.flatnonzero(np.diff(p[o]) != 0), len(y) - 1]
    return o, y[o].astype(float), last


def wap(pr, w: np.ndarray) -> float:
    """Weighted average precision; w holds the weight of each original row in this resample (bincount)."""
    o, yy, last = pr
    ww = w[o]
    tp = np.cumsum(ww * yy)[last]
    den = np.cumsum(ww)[last]
    total = tp[-1]
    if total <= 0:
        raise ValueError("zero bootstrap positives")
    return float(np.sum(np.diff(np.r_[0.0, tp]) * np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)) / total)


def interval(v: np.ndarray, lo=2.5, hi=97.5) -> list[float]:
    return [rnd(np.quantile(v, lo / 100)), rnd(np.quantile(v, hi / 100))]


def load_split(seed: int) -> dict:
    z = np.load(RES / f"localization_seed{seed}.npz", allow_pickle=True)
    y = z["y"].astype(int)
    rows = z["validation_rows"].astype(np.int64)
    assert len(y) == len(rows) == 9000, (seed, len(y), len(rows))
    out = {"seed": seed, "y": y, "rows": rows, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    for arm in ["C_local", "B_local"]:
        for cond in ["S0"] + COPIES:
            out[f"{arm}__{cond}"] = z[f"{arm}__{cond}"].astype(float)
    return out


def cond_preps(split: dict, cond: str):
    return {
        "C": prep(split["y"], split[f"C_local__{cond}"]),
        "B": prep(split["y"], split[f"B_local__{cond}"]),
    }


def point_ap(split: dict, cond: str, arm: str) -> float:
    from sklearn.metrics import average_precision_score as AP
    return float(AP(split["y"], split[f"{arm}__{cond}"]))


def per_split_bootstrap(split: dict, preps_by_cond: dict, rng: np.random.Generator):
    """Draw 1000 stratified within-split resamples by label and return the (boot_S0, boot_S1_30) arrays."""
    y = split["y"]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = len(y)
    b_s0 = np.empty(N_BOOT)
    b_s130 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        w = np.bincount(ix, minlength=n).astype(float)
        p = preps_by_cond["S0"]
        b_s0[b] = wap(p["C"], w) - wap(p["B"], w)
        b_s130[b] = np.mean([wap(preps_by_cond[c]["C"], w) - wap(preps_by_cond[c]["B"], w) for c in COPIES])
    return b_s0, b_s130


def main():
    t0 = _dt.datetime.now().astimezone()
    splits = [load_split(s) for s in SEEDS]
    rng = np.random.default_rng(BOOT_SEED)

    per_split = []
    preps = []
    point = {"S0": [], "S1_30": []}
    for sp in splits:
        pb = {c: cond_preps(sp, c) for c in ["S0"] + COPIES}
        preps.append(pb)
        ap_c = {c: point_ap(sp, c, "C_local") for c in ["S0"] + COPIES}
        ap_b = {c: point_ap(sp, c, "B_local") for c in ["S0"] + COPIES}
        s0 = ap_c["S0"] - ap_b["S0"]
        s130 = float(np.mean([ap_c[c] - ap_b[c] for c in COPIES]))
        point["S0"].append(s0)
        point["S1_30"].append(s130)
        b_s0, b_s130 = per_split_bootstrap(sp, pb, rng)
        per_split.append({
            "split_seed": sp["seed"],
            "n_validation_rows": int(len(sp["y"])),
            "n_pos": sp["n_pos"],
            "n_neg": sp["n_neg"],
            "point_estimate": {
                "C_local_AP_S0": rnd(ap_c["S0"]), "B_local_AP_S0": rnd(ap_b["S0"]), "delta_S0": rnd(s0),
                "C_local_AP_S1_30_mean": rnd(np.mean([ap_c[c] for c in COPIES])),
                "B_local_AP_S1_30_mean": rnd(np.mean([ap_b[c] for c in COPIES])),
                "delta_S1_30": rnd(s130),
                "per_copy_delta_S1_30": {c: rnd(ap_c[c] - ap_b[c]) for c in COPIES},
            },
            "bootstrap_within_split": {
                "n_bootstrap": N_BOOT, "seed": BOOT_SEED,
                "resample": "within split, stratified by label (pos/neg), fixed ens3 models",
                "delta_S0": {"mean": rnd(b_s0.mean()), "sd": rnd(b_s0.std(ddof=1)),
                             "ci95": interval(b_s0),
                             "frac_negative": rnd((b_s0 < 0).mean())},
                "delta_S1_30": {"mean": rnd(b_s130.mean()), "sd": rnd(b_s130.std(ddof=1)),
                                "ci95": interval(b_s130),
                                "frac_negative": rnd((b_s130 < 0).mean())},
            },
        })

    point_mean = {k: rnd(np.mean(v)) for k, v in point.items()}

    # ---- interval for the equal-weight mean over five splits: the A1 union-weight scheme ----
    # the five validation splits share many original rows; A1 takes the union of unique original row indices as the sampling unit and shares weight across repeats.
    union = np.unique(np.concatenate([sp["rows"] for sp in splits]))
    uy = np.empty(len(union), dtype=int)
    mappings = []
    for sp in splits:
        m = np.searchsorted(union, sp["rows"])
        uy[m] = sp["y"]
        mappings.append(m)
    upos = np.flatnonzero(uy == 1)
    uneg = np.flatnonzero(uy == 0)
    pooled_union = {"S0": np.empty(N_BOOT), "S1_30": np.empty(N_BOOT)}
    rng2 = np.random.default_rng(BOOT_SEED)
    for b in range(N_BOOT):
        ix = np.r_[rng2.choice(upos, len(upos), replace=True), rng2.choice(uneg, len(uneg), replace=True)]
        wu = np.bincount(ix, minlength=len(union)).astype(float)
        ds0, ds130 = [], []
        for pb, m in zip(preps, mappings):
            w = wu[m]
            ds0.append(wap(pb["S0"]["C"], w) - wap(pb["S0"]["B"], w))
            ds130.append(np.mean([wap(pb[c]["C"], w) - wap(pb[c]["B"], w) for c in COPIES]))
        pooled_union["S0"][b] = np.mean(ds0)
        pooled_union["S1_30"][b] = np.mean(ds130)

    # ---- sensitivity: resample within each split independently, then average the five splits ----
    pooled_indep = {"S0": np.empty(N_BOOT), "S1_30": np.empty(N_BOOT)}
    rng3 = np.random.default_rng(BOOT_SEED + 1)
    draws = {"S0": np.empty((len(splits), N_BOOT)), "S1_30": np.empty((len(splits), N_BOOT))}
    for i, (sp, pb) in enumerate(zip(splits, preps)):
        b_s0, b_s130 = per_split_bootstrap(sp, pb, rng3)
        draws["S0"][i] = b_s0
        draws["S1_30"][i] = b_s130
    pooled_indep["S0"] = draws["S0"].mean(axis=0)
    pooled_indep["S1_30"] = draws["S1_30"].mean(axis=0)

    pooled = {}
    for cond, key in [("S0", "S0"), ("S1_30", "S1_30")]:
        bu = pooled_union[key]
        bi = pooled_indep[key]
        pooled[key] = {
            "point_estimate_five_split_mean_delta": point_mean[key],
            "per_split_point_estimates": [rnd(v) for v in point[key]],
            "n_splits_negative": int(sum(1 for v in point[key] if v < 0)),
            "n_splits_positive": int(sum(1 for v in point[key] if v > 0)),
            "union_weight_scheme_A1": {
                "description": "sampling unit is the union of unique original row indices across the five validation splits, stratified by label, overlapping rows share weight; weighted AP differences are averaged over the five splits",
                "n_unique_rows_in_union": int(len(union)),
                "n_pos_in_union": int(len(upos)),
                "n_neg_in_union": int(len(uneg)),
                "n_bootstrap": N_BOOT, "seed": BOOT_SEED,
                "mean_of_draws": rnd(bu.mean()), "sd_of_draws": rnd(bu.std(ddof=1)),
                "ci95": interval(bu),
                "frac_draws_negative": rnd((bu < 0).mean()),
            },
            "independent_within_split_sensitivity": {
                "description": "resample within each split independently by label, then average the five splits (ignores within-row correlation across splits)",
                "n_bootstrap": N_BOOT, "seed": BOOT_SEED + 1,
                "mean_of_draws": rnd(bi.mean()), "sd_of_draws": rnd(bi.std(ddof=1)),
                "ci95": interval(bi),
                "frac_draws_negative": rnd((bi < 0).mean()),
            },
        }

    payload = {
        "generated_at": t0.isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "alpha": 0.25,
        "design": "same actual per-column deletion count in each selected column; positive-targeted vs row-class-blind; other cells unchanged",
        "estimand": "AP difference C_local - B_local; for S1_30 the three stress replicates are averaged (as in the manuscript table)",
        "description": (
            "R9 supplement/B3: 1000-draw row-resampling conditional intervals for the localization ablation. Intervals are conditional on the fixed ens3 models and the fixed splits,"
            "and cover neither retraining, reselection nor split randomness; the scheme matches the main conditional intervals (A1 union weights)."
        ),
        "sources": [
            *[{"path": str(RES / f"localization_seed{s}.npz"), "sha256": sha256(RES / f"localization_seed{s}.npz"),
               "role": f"per-row out-of-fold scores of local C/B for split {s}"} for s in SEEDS],
            {"path": str(RES / "localization.json"), "sha256": sha256(RES / "localization.json"),
             "role": "design and point estimates of the localization ablation (manuscript table)"},
        ],
        "per_split": per_split,
        "pooled_five_splits": pooled,
    }
    out = OUT_DIR / "r9_localization_intervals.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print("WROTE", out)
    print("\n=== point estimates ===")
    for r in per_split:
        print(r["split_seed"], "S0", r["point_estimate"]["delta_S0"],
              "CI", r["bootstrap_within_split"]["delta_S0"]["ci95"],
              "S1_30", r["point_estimate"]["delta_S1_30"],
              "CI", r["bootstrap_within_split"]["delta_S1_30"]["ci95"])
    print("\n=== pooled ===")
    for k, v in pooled.items():
        print(k, "point", v["point_estimate_five_split_mean_delta"],
              "A1 union CI", v["union_weight_scheme_A1"]["ci95"],
              "frac<0", v["union_weight_scheme_A1"]["frac_draws_negative"],
              "| indep CI", v["independent_within_split_sensitivity"]["ci95"])


if __name__ == "__main__":
    main()
