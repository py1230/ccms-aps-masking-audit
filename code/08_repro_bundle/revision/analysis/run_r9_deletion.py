#!/usr/bin/env python
"""R9 / B1: diagnostics of the *actual* deletion dose of the APS fixed-alpha = 0.25 comparison (no model is refitted).

Reads only stored results and the original CSV:
  - the APS training CSV in the package data directory (read with the packaged src/aps_io.py)
  - train_rows of results/aps_seed{11,22,33,44,55}.npz (to recompute q and the class counts of that split)
  - splits[i].deletion_diagnostics in results/aps_reconstruction.json
      (three C entries and three arm='B' entries per split)

Output: revision/analysis/r9_deletion_diagnostics.json
Every number comes from those files; nothing is invented.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys

sys.dont_write_bytecode = True  # do not write __pycache__ into the archived package
import datetime as _dt
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2] / "CCMS论文修订完整资料包_2026-11-01_192149"
RES = PKG / "02_本轮代码与结果/results"
SRC = PKG / "03_原始项目归档/CCMS核查_2026-10-31/src"
TRAIN_CSV = PKG / "04_实际使用的数据/aps/aps_failure_training_set.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
from aps_io import load_aps_csv, split_features  # noqa: E402
from ccms import estimate_class_rates, class_blind_rates, class_conditional_rates, _qmax  # noqa: E402

ALPHA = 0.25
SEEDS = [11, 22, 33, 44, 55]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def f(x):
    """Convert a numpy scalar to a Python float so that it can be serialised to JSON."""
    return float(x)


def rnd(x, n=6):
    return float(np.round(float(x), n))


def _ratio_block(c: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict:
    ok = m & (b > 0)
    if ok.sum() == 0:
        return {"n_columns": 0}
    ratio = c[ok] / b[ok]
    dev = np.abs(ratio - 1.0)
    return {
        "n_columns": int(ok.sum()),
        "mean": rnd(ratio.mean()),
        "median": rnd(np.median(ratio)),
        "sd": rnd(ratio.std(ddof=1)),
        "min": rnd(ratio.min()),
        "max": rnd(ratio.max()),
        "n_abs_ratio_minus_1_gt_0.5": int((dev > 0.5).sum()),
        "n_abs_ratio_minus_1_gt_0.1": int((dev > 0.1).sum()),
        "columns_abs_ratio_minus_1_gt_0.5": [int(j) for j in np.flatnonzero(ok & (np.abs(c / np.where(b > 0, b, 1.0) - 1.0) > 0.5))],
    }


def col_ratio_stats(c: np.ndarray, b: np.ndarray, expected: np.ndarray) -> dict:
    """Compare the distributions of C_j and B_j column by column.

    In 74 of the 170 columns the expected training deletion count is tiny (together below 0.1% of the budget) and the ratio is pure counting noise;
    so both the all-column view and the stable-column view (expected deletions >= 20) are reported; only the latter is interpretable.
    """
    low = expected < 20.0
    return {
        "n_columns_total": int(len(c)),
        "n_columns_stable_expected_ge_20": int((~low).sum()),
        "n_columns_low_expected_lt_20": int(low.sum()),
        "low_expected_columns_expected_sum": rnd(expected[low].sum()),
        "low_expected_share_of_total_budget": rnd(expected[low].sum() / expected.sum()),
        "all_columns": _ratio_block(c, b, np.ones(len(c), dtype=bool)),
        "stable_columns_expected_ge_20": _ratio_block(c, b, ~low),
    }


def _z_block(actual, expected, var, m) -> dict:
    ok = m & (var > 0)
    if ok.sum() == 0:
        return {"n_columns": 0}
    z = (actual[ok] - expected[ok]) / np.sqrt(var[ok])
    return {
        "n_columns": int(ok.sum()),
        "max_abs_z": rnd(np.abs(z).max()),
        "rmse_z": rnd(np.sqrt(np.mean(z ** 2))),
        "n_abs_z_gt_3": int((np.abs(z) > 3).sum()),
        "argmax_abs_z_column": int(np.flatnonzero(ok)[int(np.argmax(np.abs(z)))]),
    }


def rel_dev(actual: np.ndarray, expected: np.ndarray, var: np.ndarray) -> dict:
    """Column-wise actual/expected reconciliation diagnostics.

    - Relative deviation rel_j = actual_j/expected_j - 1; it explodes for small counts, so two views are reported.
    - Standardised residual z_j = (actual_j - expected_j)/sqrt(var_j), where var_j is the binomial variance of independent Bernoulli masking
      (summed over classes for C; visible cells * b * (1 - b) for B). This is the correct scale for a per-column reconciliation.
    """
    ok = expected > 0
    rel = actual[ok] / expected[ok] - 1.0
    ar = np.abs(rel)
    stable = expected >= 20.0
    sel = ok & stable
    return {
        "n_columns_defined": int(ok.sum()),
        "n_columns_expected_zero": int((~ok).sum()),
        "max_abs_rel_all_defined": rnd(ar.max()),
        "mean_abs_rel_all_defined": rnd(ar.mean()),
        "rmse_rel_all_defined": rnd(np.sqrt(np.mean(rel ** 2))),
        "argmax_abs_rel_column": int(np.flatnonzero(ok)[int(np.argmax(ar))]),
        "expected_ge_20": {
            "n_columns": int(sel.sum()),
            "max_abs_rel": rnd(np.abs(actual[sel] / expected[sel] - 1).max()) if sel.any() else None,
            "rmse_rel": rnd(np.sqrt(np.mean((actual[sel] / expected[sel] - 1) ** 2))) if sel.any() else None,
        },
        "standardized_z": _z_block(actual, expected, var, ok),
        "max_abs_abs_dev": rnd(np.abs(actual - expected).max()),
        "sum_actual": int(actual.sum()),
        "sum_expected": rnd(expected.sum()),
        "total_ratio_actual_over_expected": rnd(actual.sum() / expected.sum()),
    }


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance 0.5*sum|p_j/p.sum - q_j/q.sum|, computed on already normalised profiles."""
    p = p.astype(float) / p.sum()
    q = q.astype(float) / q.sum()
    return rnd(0.5 * np.abs(p - q).sum())


def topk_share(per_col: np.ndarray, k: int, mask: np.ndarray | None = None) -> dict:
    """Take the k largest entries of per_col and return their share of the total (or of a given total)."""
    order = np.argsort(-per_col, kind="stable")[:k]
    tot = per_col.sum() if mask is None else mask
    return {
        "k": k,
        "columns": [int(j) for j in order],
        "share": rnd(per_col[order].sum() / tot),
        "abs": int(per_col[order].sum()),
    }


def parse_one_split(split: dict, X: np.ndarray, y: np.ndarray) -> dict:
    seed = int(split["seed"])
    z = np.load(RES / f"aps_seed{seed}.npz", allow_pickle=True)
    tr = z["train_rows"]
    x = X[tr]
    yy = y[tr]

    q, counts, _ = estimate_class_rates(x, yy)
    qmax = _qmax(q)
    n0, n1 = int(counts[0]), int(counts[1])
    n = n0 + n1
    visible = (~np.isnan(x)).sum(axis=0)
    vis_c = np.array([visible * (yy[None, :] == 0).sum() / n, visible * (yy[None, :] == 1).sum() / n])
    # more robust: count visible cells directly by class
    vis_c = np.array([np.sum(~np.isnan(x[yy == 0]), axis=0), np.sum(~np.isnan(x[yy == 1]), axis=0)])
    q0 = q[0]
    q1 = q[1]
    # independently recompute the expected deletion vectors (C and B must agree column by column; see the assert in run_supplement.py)
    exp_c = ALPHA * (counts[:, None] * (qmax[None, :] - np.where(np.isnan(q), 0.0, q))).sum(axis=0)
    bvec = class_blind_rates(q, counts, ALPHA)
    a_rates = class_conditional_rates(q, ALPHA)  # 2x170 additional masking probabilities
    exp_b = bvec * visible
    # column-wise binomial variance under independent Bernoulli masking: C sums over classes, B applies one probability to all visible cells
    var_c = (vis_c * a_rates * (1.0 - a_rates)).sum(axis=0)
    var_b = visible * bvec * (1.0 - bvec)
    assert np.allclose(exp_c, exp_b, atol=1e-8), "C and B expected budgets differ, contradicting the assertion in the archived script"

    diags = split["deletion_diagnostics"]
    C = sorted([d for d in diags if d.get("arm") != "B"], key=lambda d: d["mask_seed"])
    B = sorted([d for d in diags if d.get("arm") == "B"], key=lambda d: d["mask_seed"])
    assert len(C) == 3 and len(B) == 3

    pairs = []
    for c_item, b_item in zip(C, B):
        assert c_item["mask_seed"] == b_item["mask_seed"]
        ms = int(c_item["mask_seed"])
        dc = np.asarray(c_item["deleted_by_class_col"], dtype=float)  # 2x170
        db = np.asarray(b_item["deleted_by_class_col"], dtype=float)  # 2x170
        actual_c = np.asarray(c_item["actual_per_col"], dtype=float)
        actual_b = np.asarray(b_item["actual_per_col"], dtype=float)
        exp_stored = np.asarray(c_item["expected_per_col"], dtype=float)
        assert np.allclose(exp_stored, exp_c, atol=1e-8), f"{seed}/{ms}: stored expectation differs from the recomputation"
        assert np.allclose(actual_c, dc.sum(0)), f"{seed}/{ms}: C column sums differ"
        assert np.allclose(actual_b, db.sum(0)), f"{seed}/{ms}: B column sums differ"
        vis_bc = np.asarray(c_item["visible_by_class_col"], dtype=float)
        assert np.allclose(vis_bc, vis_c)

        C_tot = dc.sum()
        B_tot = db.sum()
        c_neg, c_pos = dc[0].sum(), dc[1].sum()
        b_neg, b_pos = db[0].sum(), db[1].sum()
        vis_neg, vis_pos = vis_c[0].sum(), vis_c[1].sum()

        # positive-class deletions expected in training (definition of the eight columns selected by expected positive deletions, Section 4.4)
        expected_pos = ALPHA * n1 * (qmax - q1)
        top8_expected_pos = np.argsort(-expected_pos, kind="stable")[:8]

        # concentration of all C deletions (by actual column-wise deletions)
        top_total = {f"top{k}": topk_share(actual_c, k) for k in (8, 10, 20)}
        # concentration of C's positive-class deletions: the 85.0%-91.7% figure of the manuscript
        pos_del = dc[1]
        top8_pos_actual = np.argsort(-pos_del, kind="stable")[:8]
        pos_total = pos_del.sum()
        share_pos_top8_expected_cols = float(pos_del[top8_expected_pos].sum() / pos_total) if pos_total > 0 else None
        share_pos_top8_actual_cols = float(pos_del[top8_pos_actual].sum() / pos_total) if pos_total > 0 else None
        # share and absolute count of these eight columns in all C deletions (the denominator is all deletions, not positive-class deletions)
        top8_default_cols = top8_expected_pos
        abs_pos_in_top8 = int(pos_del[top8_default_cols].sum())
        share_of_C_total_in_top8 = float(abs_pos_in_top8 / C_tot)
        # positive-class and total deletions of B in the same eight columns
        abs_pos_in_top8_B = int(db[1][top8_default_cols].sum())
        share_of_B_total_in_top8 = float(db[:, top8_default_cols].sum() / B_tot)
        # deletion rate of each class within these eight columns (relative to the visible cells of that class)
        c_rate_top8_pos = float(pos_del[top8_default_cols].sum() / vis_c[1][top8_default_cols].sum())
        b_rate_top8_pos = float(db[1][top8_default_cols].sum() / vis_c[1][top8_default_cols].sum())
        c_rate_top8_neg = float(dc[0][top8_default_cols].sum() / vis_c[0][top8_default_cols].sum())
        b_rate_top8_neg = float(db[0][top8_default_cols].sum() / vis_c[0][top8_default_cols].sum())

        pair = {
            "mask_seed": ms,
            "class_counts": {"n0_neg": n0, "n1_pos": n1, "n_total": n},
            "C": {
                "total": int(C_tot),
                "by_class": {"neg": int(c_neg), "pos": int(c_pos)},
                "class_share": {"neg": rnd(c_neg / C_tot), "pos": rnd(c_pos / C_tot)},
                "visible_by_class": {"neg": int(vis_neg), "pos": int(vis_pos)},
                "per_class_deletion_rate": {"neg": rnd(c_neg / vis_neg), "pos": rnd(c_pos / vis_pos)},
                "top_total_concentration": top_total,
                "top8_pos_share_of_pos_deletions": rnd(share_pos_top8_actual_cols),
                "top8_pos_columns": [int(j) for j in top8_pos_actual],
                "top8_expected_pos_columns": [int(j) for j in top8_expected_pos],
                "top8_pos_share_expected_cols": rnd(share_pos_top8_expected_cols),
                "top8_pos_abs_deletions": abs_pos_in_top8,
                "top8_pos_share_of_C_total": rnd(share_of_C_total_in_top8),
                "top8_expected_abs_pos_deletion_sum": rnd(float(expected_pos[top8_expected_pos].sum())),
                "rel_dev_actual_vs_expected": rel_dev(actual_c, exp_c, var_c),
                "top8_class_deletion_rate": {"pos_C": rnd(c_rate_top8_pos), "pos_B": rnd(b_rate_top8_pos),
                                             "neg_C": rnd(c_rate_top8_neg), "neg_B": rnd(b_rate_top8_neg)},
            },
            "B": {
                "total": int(B_tot),
                "by_class": {"neg": int(b_neg), "pos": int(b_pos)},
                "class_share": {"neg": rnd(b_neg / B_tot), "pos": rnd(b_pos / B_tot)},
                "visible_by_class": {"neg": int(vis_neg), "pos": int(vis_pos)},
                "per_class_deletion_rate": {"neg": rnd(b_neg / vis_neg), "pos": rnd(b_pos / vis_pos)},
                "rel_dev_actual_vs_expected": rel_dev(actual_b, exp_b, var_b),
                "top8_pos_abs_deletions_same_expected_cols": abs_pos_in_top8_B,
                "top8_share_of_B_total_same_cols": rnd(share_of_B_total_in_top8),
            },
            "pair_balance": {
                "total_ratio_C_over_B": rnd(C_tot / B_tot),
                "total_diff_C_minus_B": int(C_tot - B_tot),
                "class_diff_C_minus_B": {"neg": int(c_neg - b_neg), "pos": int(c_pos - b_pos)},
                "pos_share_ratio_C_over_B": rnd((c_pos / C_tot) / (b_pos / B_tot)),
                "tv_column_profile": tv(actual_c, actual_b),
                "tv_class_profile": {
                    "neg": tv(dc[0], db[0]) if dc[0].sum() > 0 and db[0].sum() > 0 else None,
                    "pos": tv(dc[1], db[1]) if dc[1].sum() > 0 and db[1].sum() > 0 else None,
                },
                "column_ratio_C_over_B": col_ratio_stats(actual_c, actual_b, exp_c),
            },
        }
        pairs.append(pair)

    # cell-level nested check of the shared-uniform protocol: C and B use the same uniforms (same mask_seed) and the target class has a > b,
    # so in a target column the C deletions of the target class should be a superset of B's; in other classes C deletes nothing while B still deletes.
    cols_pos_target = np.flatnonzero(q0 > q1)
    cols_neg_target = np.flatnonzero(q1 > q0)
    nest = []
    for c_item, b_item in zip(C, B):
        dcc = np.asarray(c_item["deleted_by_class_col"], dtype=int)
        dbb = np.asarray(b_item["deleted_by_class_col"], dtype=int)
        nest.append({
            "mask_seed": int(c_item["mask_seed"]),
            "C_neg_superset_of_B_neg_on_neg_target_columns": bool(np.all(dcc[0][cols_neg_target] >= dbb[0][cols_neg_target])),
            "C_pos_superset_of_B_pos_on_pos_target_columns": bool(np.all(dcc[1][cols_pos_target] >= dbb[1][cols_pos_target])),
            "C_neg_deletions_on_pos_target_columns": int(dcc[0][cols_pos_target].sum()),
            "C_pos_deletions_on_neg_target_columns": int(dcc[1][cols_neg_target].sum()),
        })
    shared_uniform_check = {
        "a_neg_gt_b_on_neg_target_columns": bool(np.all(a_rates[0][cols_neg_target] > bvec[cols_neg_target])),
        "a_pos_gt_b_on_pos_target_columns": bool(np.all(a_rates[1][cols_pos_target] > bvec[cols_pos_target])),
        "per_mask_seed": nest,
    }

    # split-level q and class counts
    cols_with_q0_gt_q1 = [int(j) for j in np.flatnonzero(q0 > q1)]
    cols_with_q1_gt_q0 = [int(j) for j in np.flatnonzero(q1 > q0)]
    cols_equal = int(np.sum(q0 == q1))
    split_out = {
        "split_seed": seed,
        "n_train": n,
        "class_counts": {"n0_neg": n0, "n1_pos": n1},
        "pos_row_fraction": rnd(n1 / n),
        "class_cell_missing_rate_overall": {"neg_q0": rnd(q[0].mean()), "pos_q1": rnd(q[1].mean())},
        "target_class_less_missing": "neg" if q[0].mean() < q[1].mean() else "pos",
        "target_class_row_share": rnd(n0 / n) if q[0].mean() < q[1].mean() else rnd(n1 / n),
        "per_column_target": {
            "n_columns_q0_gt_q1_pos_targeted": len(cols_with_q0_gt_q1),
            "n_columns_q1_gt_q0_neg_targeted": len(cols_with_q1_gt_q0),
            "n_columns_equal": cols_equal,
            "columns_q0_gt_q1": cols_with_q0_gt_q1,
        },
        "shared_uniform_nesting_check": shared_uniform_check,
        "expected_budget": {
            "expected_total": rnd(exp_c.sum()),
            "expected_pos_class": rnd(ALPHA * n1 * np.clip(qmax - q1, 0, None).sum()),
            "expected_neg_class": rnd(ALPHA * n0 * np.clip(qmax - q0, 0, None).sum()),
        },
        "pairs": pairs,
    }
    return split_out


def agg_pooled(splits: list[dict]) -> dict:
    def collect(fn):
        return [fn(sp) for sp in splits]

    # average over the three masking seeds within a split, then take the mean and range over the five splits
    def split_mean(sp, fn):
        return float(np.mean([fn(p) for p in sp["pairs"]]))

    keys = {
        "C_total": lambda p: p["C"]["total"],
        "B_total": lambda p: p["B"]["total"],
        "total_ratio_C_over_B": lambda p: p["pair_balance"]["total_ratio_C_over_B"],
        "C_neg_share": lambda p: p["C"]["class_share"]["neg"],
        "C_pos_share": lambda p: p["C"]["class_share"]["pos"],
        "B_neg_share": lambda p: p["B"]["class_share"]["neg"],
        "B_pos_share": lambda p: p["B"]["class_share"]["pos"],
        "C_neg_rate": lambda p: p["C"]["per_class_deletion_rate"]["neg"],
        "C_pos_rate": lambda p: p["C"]["per_class_deletion_rate"]["pos"],
        "B_neg_rate": lambda p: p["B"]["per_class_deletion_rate"]["neg"],
        "B_pos_rate": lambda p: p["B"]["per_class_deletion_rate"]["pos"],
        "C_pos_rate_over_B_pos_rate": lambda p: p["C"]["per_class_deletion_rate"]["pos"] / p["B"]["per_class_deletion_rate"]["pos"],
        "C_neg_rate_over_B_neg_rate": lambda p: p["C"]["per_class_deletion_rate"]["neg"] / p["B"]["per_class_deletion_rate"]["neg"],
        "top8_pos_share_of_pos_deletions": lambda p: p["C"]["top8_pos_share_of_pos_deletions"],
        "top8_pos_share_of_C_total": lambda p: p["C"]["top8_pos_share_of_C_total"],
        "top8_pos_abs_deletions": lambda p: p["C"]["top8_pos_abs_deletions"],
        "top8_total_share_of_C_total": lambda p: p["C"]["top_total_concentration"]["top8"]["share"],
        "top10_total_share_of_C_total": lambda p: p["C"]["top_total_concentration"]["top10"]["share"],
        "top20_total_share_of_C_total": lambda p: p["C"]["top_total_concentration"]["top20"]["share"],
        "tv_column_profile": lambda p: p["pair_balance"]["tv_column_profile"],
        "tv_neg_profile": lambda p: p["pair_balance"]["tv_class_profile"]["neg"],
        "tv_pos_profile": lambda p: p["pair_balance"]["tv_class_profile"]["pos"],
        "col_ratio_mean_all": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["all_columns"]["mean"],
        "col_ratio_sd_all": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["all_columns"]["sd"],
        "col_ratio_n_gt_0.5_all": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["all_columns"]["n_abs_ratio_minus_1_gt_0.5"],
        "col_ratio_mean_stable": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["stable_columns_expected_ge_20"]["mean"],
        "col_ratio_sd_stable": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["stable_columns_expected_ge_20"]["sd"],
        "col_ratio_min_stable": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["stable_columns_expected_ge_20"]["min"],
        "col_ratio_max_stable": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["stable_columns_expected_ge_20"]["max"],
        "col_ratio_n_gt_0.5_stable": lambda p: p["pair_balance"]["column_ratio_C_over_B"]["stable_columns_expected_ge_20"]["n_abs_ratio_minus_1_gt_0.5"],
        "C_rel_dev_max_abs_stable": lambda p: p["C"]["rel_dev_actual_vs_expected"]["expected_ge_20"]["max_abs_rel"],
        "C_rel_dev_rmse_stable": lambda p: p["C"]["rel_dev_actual_vs_expected"]["expected_ge_20"]["rmse_rel"],
        "B_rel_dev_max_abs_stable": lambda p: p["B"]["rel_dev_actual_vs_expected"]["expected_ge_20"]["max_abs_rel"],
        "B_rel_dev_rmse_stable": lambda p: p["B"]["rel_dev_actual_vs_expected"]["expected_ge_20"]["rmse_rel"],
        "C_z_max": lambda p: p["C"]["rel_dev_actual_vs_expected"]["standardized_z"]["max_abs_z"],
        "C_z_rmse": lambda p: p["C"]["rel_dev_actual_vs_expected"]["standardized_z"]["rmse_z"],
        "B_z_max": lambda p: p["B"]["rel_dev_actual_vs_expected"]["standardized_z"]["max_abs_z"],
        "B_z_rmse": lambda p: p["B"]["rel_dev_actual_vs_expected"]["standardized_z"]["rmse_z"],
        "C_minus_B_pos": lambda p: p["pair_balance"]["class_diff_C_minus_B"]["pos"],
        "C_minus_B_neg": lambda p: p["pair_balance"]["class_diff_C_minus_B"]["neg"],
    }
    out = {}
    for name, fn in keys.items():
        per_split_means = np.array([split_mean(sp, fn) for sp in splits], dtype=float)
        all_pairs = np.array([fn(p) for sp in splits for p in sp["pairs"]], dtype=float)
        out[name] = {
            "per_split_mean_over_3_mask_seeds": [rnd(v) for v in per_split_means],
            "five_split_mean": rnd(per_split_means.mean()),
            "five_split_min": rnd(per_split_means.min()),
            "five_split_max": rnd(per_split_means.max()),
            "all_15_pairs_min": rnd(all_pairs.min()),
            "all_15_pairs_max": rnd(all_pairs.max()),
        }
    return out


def main():
    t0 = _dt.datetime.now().astimezone()
    df, meta = load_aps_csv(TRAIN_CSV)
    X, y = split_features(df)
    miss = np.isnan(X)
    full = {
        "n_rows": int(len(y)),
        "n_columns": int(X.shape[1]),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "pos_row_fraction": rnd(y.mean()),
        "pos_cell_missing_rate": rnd(miss[y == 1].mean()),
        "neg_cell_missing_rate": rnd(miss[y == 0].mean()),
        "majority_row_fraction": rnd((y == 0).mean()),
        "target_class_less_missing": "neg" if miss[y == 0].mean() < miss[y == 1].mean() else "pos",
        "pos_visible_cell_fraction": rnd((~miss[y == 1]).sum() / (~miss).sum()),
        "neg_visible_cell_fraction": rnd((~miss[y == 0]).sum() / (~miss).sum()),
    }
    assert full["n_rows"] == 60000 and full["n_pos"] == 1000 and full["n_neg"] == 59000, "APS training counts disagree with the lead cross-check"
    assert abs(full["pos_cell_missing_rate"] - 0.171488) < 5e-4
    assert abs(full["neg_cell_missing_rate"] - 0.081841) < 5e-4

    recon = json.loads((RES / "aps_reconstruction.json").read_text())
    splits = [parse_one_split(sp, X, y) for sp in recon["splits"]]
    pooled = agg_pooled(splits)

    payload = {
        "generated_at": t0.isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "alpha": ALPHA,
        "description": (
            "R9/B1: actual deletion-dose diagnostics of the APS fixed alpha=0.25 comparison. The actual per-column and per-class deletion counts of C and B"
            "come from deletion_diagnostics in aps_reconstruction.json; q and the class counts are recomputed independently from "
            "train_rows in aps_seed*.npz and the original training CSV. No model was refitted."
        ),
        "sources": [
            {"path": str(TRAIN_CSV), "sha256": sha256(TRAIN_CSV), "role": "original APS training CSV"},
            {"path": str(RES / "aps_reconstruction.json"), "sha256": sha256(RES / "aps_reconstruction.json"),
             "role": "source of deletion_diagnostics"},
            *[{"path": str(RES / f"aps_seed{s}.npz"), "sha256": sha256(RES / f"aps_seed{s}.npz"),
               "role": f"train_rows / val_rows of split {s}"} for s in SEEDS],
            {"path": str(SRC / "aps_io.py"), "sha256": sha256(SRC / "aps_io.py"), "role": "load_aps_csv / split_features"},
            {"path": str(SRC / "ccms.py"), "sha256": sha256(SRC / "ccms.py"), "role": "estimate_class_rates / class_blind_rates / _qmax"},
        ],
        "crosscheck_full_training_set": full,
        "splits": splits,
        "pooled_five_splits": pooled,
        "verify_vs_archived_mechanism_json": verify_against_archived(splits),
    }
    out = OUT_DIR / "r9_deletion_diagnostics.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print("WROTE", out)
    print(json.dumps({k: v["five_split_mean"] for k, v in pooled.items()}, ensure_ascii=False, indent=1))
    for sp in splits:
        print("split", sp["split_seed"], "q0", sp["class_cell_missing_rate_overall"]["neg_q0"],
              "q1", sp["class_cell_missing_rate_overall"]["pos_q1"], "target", sp["target_class_less_missing"],
              "ncols_pos_targeted", sp["per_column_target"]["n_columns_q0_gt_q1_pos_targeted"])


def verify_against_archived(splits: list[dict]) -> dict:
    """Cross-check against the quantity of the same name in the archived mechanism_alpha025.json (produced by the packaged summarize.py)."""
    path = RES / "mechanism_alpha025.json"
    if not path.exists():
        return {"available": False}
    arch = json.loads(path.read_text())
    amap = {int(r["seed"]): r for r in arch["splits"]}
    checks = []
    for sp in splits:
        a = amap[sp["split_seed"]]
        shares_mine = [p["C"]["top8_pos_share_of_pos_deletions"] for p in sp["pairs"]]
        shares_arch = [float(v) for v in a["pos_deletion_top8_share"]]
        cdel_mine = [[p["C"]["by_class"]["neg"], p["C"]["by_class"]["pos"]] for p in sp["pairs"]]
        bdel_mine = [[p["B"]["by_class"]["neg"], p["B"]["by_class"]["pos"]] for p in sp["pairs"]]
        checks.append({
            "split_seed": sp["split_seed"],
            "max_abs_diff_top8_pos_share": rnd(np.max(np.abs(np.array(shares_mine) - np.array(shares_arch)))),
            "C_by_class_equal": bool(np.array_equal(np.array(cdel_mine), np.array(a["C_deleted"]))),
            "B_by_class_equal": bool(np.array_equal(np.array(bdel_mine), np.array(a["B_deleted"]))),
        })
    return {"available": True, "path": str(path), "sha256": sha256(path), "checks": checks}



if __name__ == "__main__":
    main()
