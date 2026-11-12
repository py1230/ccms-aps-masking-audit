#!/usr/bin/env python3
"""
Recompute conditional SECOM intervals from archived fold-level predictions.

This is a prediction-level reanalysis only:
- no raw SECOM feature data are read;
- no model is retrained;
- the selected alpha values and saved predictions remain fixed.

Bootstrap protocol:
1. Each outer test fold is a separate resampling stratum.
2. Within each fold, positive and negative rows are sampled with replacement,
   preserving that fold's observed class counts.
3. The same row-count weights are shared by C and B and by all three S1_30
   evaluation-mask repeats, preserving paired comparisons.
4. AP is calculated separately for each evaluation-mask repeat; the three
   S1_30 AP values are averaged within fold.
5. The five fold-level differences are averaged for each bootstrap replicate.
6. Percentile 95% intervals are calculated from 20,000 replicates.

The intervals are conditional on the archived fitted models, masks, and
predictions. They do not include retraining, alpha reselection, or dependence
induced by overlapping outer training sets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_ap(y: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order].astype(np.float64)
    s_sorted = scores[order]
    group_ends = np.r_[np.flatnonzero(np.diff(s_sorted)), len(y_sorted) - 1]
    return order, y_sorted, group_ends


def weighted_ap_from_counts(counts: np.ndarray, prepared) -> float:
    order, y_sorted, group_ends = prepared
    w_sorted = counts[order].astype(np.float64)
    cumulative_tp = np.cumsum(y_sorted * w_sorted)
    cumulative_n = np.cumsum(w_sorted)
    total_positive = cumulative_tp[-1]
    if total_positive <= 0:
        raise ValueError("Bootstrap replicate contains no positive weight.")
    previous_tp = np.r_[0.0, cumulative_tp[group_ends[:-1]]]
    delta_tp = cumulative_tp[group_ends] - previous_tp
    valid = cumulative_n[group_ends] > 0
    precision = cumulative_tp[group_ends[valid]] / cumulative_n[group_ends[valid]]
    return float(np.sum((delta_tp[valid] / total_positive) * precision))


def load_npz_from_zip(zf: zipfile.ZipFile, member: str):
    return np.load(io.BytesIO(zf.read(member)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm-zip", required=True, help="Path to ESM_2.zip")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    root = "ESM_2/08_repro_bundle/02_本轮代码与结果/results"
    fold_arrays = []
    source_hashes = []
    test_row_ids = []

    with zipfile.ZipFile(args.esm_zip) as zf:
        for fi in range(5):
            member = f"{root}/secom_fold{fi}.npz"
            raw = zf.read(member)
            source_hashes.append({
                "path": member,
                "n_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            })
            a = np.load(io.BytesIO(raw))
            train = set(map(int, a["train_rows"]))
            val = set(map(int, a["val_rows"]))
            test = set(map(int, a["test_rows"]))
            if train & val or train & test or val & test:
                raise ValueError(f"Overlapping train/validation/test rows in fold {fi}.")
            test_row_ids.extend(map(int, a["test_rows"]))
            fold_arrays.append({k: a[k] for k in a.files})

        summary_member = f"{root}/secom_summary.json"
        archived_summary = json.loads(zf.read(summary_member))

    if len(test_row_ids) != len(set(test_row_ids)):
        raise ValueError("Outer test folds are not disjoint.")
    if sorted(test_row_ids) != list(range(len(test_row_ids))):
        raise ValueError("Outer test folds do not form a complete row partition.")

    conditions = {"S0": [None], "S1_30": ["101", "202", "303"]}
    pairs = {
        "fixed_alpha_0.25": ("C025", "B025"),
        "validation_selected": ("C_selected", "B_selected"),
    }

    prepared = {}
    point_by_fold = {}
    for fi, fold in enumerate(fold_arrays):
        y = fold["y_test"].astype(int)
        for condition, repeats in conditions.items():
            for arm in ["C025", "B025", "C_selected", "B_selected"]:
                pred_prepared = []
                point_aps = []
                for rep in repeats:
                    key = f"{arm}__test__{condition}" if rep is None else f"{arm}__test__{condition}_{rep}"
                    scores = fold[key].astype(float)
                    pred_prepared.append(prepare_ap(y, scores))
                    point_aps.append(float(average_precision_score(y, scores)))
                prepared[(fi, condition, arm)] = pred_prepared
                point_by_fold[(fi, condition, arm)] = float(np.mean(point_aps))

    point_results = {}
    for condition in conditions:
        point_results[condition] = {}
        for pair_name, (arm_a, arm_b) in pairs.items():
            per_fold = [
                point_by_fold[(fi, condition, arm_a)] - point_by_fold[(fi, condition, arm_b)]
                for fi in range(5)
            ]
            point_results[condition][pair_name] = {
                "delta_mean": float(np.mean(per_fold)),
                "per_fold_delta": [float(x) for x in per_fold],
            }

    # Confirm the recomputed point estimates against the archived summary.
    archived_point_checks = {}
    for condition in conditions:
        archived_key = "fixed_C-B" if "fixed_C-B" in archived_summary["paired"] else "fixed_C-B"
        for pair_name in pairs:
            key = "fixed_C-B" if pair_name == "fixed_alpha_0.25" else "selected_C-B"
            expected = float(archived_summary["paired"][key][condition]["delta_mean"])
            observed = point_results[condition][pair_name]["delta_mean"]
            archived_point_checks[f"{condition}/{pair_name}"] = {
                "archived_delta_mean": expected,
                "recomputed_delta_mean": observed,
                "absolute_difference": abs(expected - observed),
            }

    rng = np.random.default_rng(args.seed)
    bootstrap = {
        f"{pair_name}/{condition}": np.empty(args.n_boot, dtype=np.float64)
        for pair_name in pairs for condition in conditions
    }

    for b in range(args.n_boot):
        fold_deltas = {key: [] for key in bootstrap}
        for fi, fold in enumerate(fold_arrays):
            y = fold["y_test"].astype(int)
            positive = np.flatnonzero(y == 1)
            negative = np.flatnonzero(y == 0)
            sampled = np.concatenate([
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ])
            counts = np.bincount(sampled, minlength=len(y))

            for condition in conditions:
                for pair_name, (arm_a, arm_b) in pairs.items():
                    ap_a = np.mean([
                        weighted_ap_from_counts(counts, pr)
                        for pr in prepared[(fi, condition, arm_a)]
                    ])
                    ap_b = np.mean([
                        weighted_ap_from_counts(counts, pr)
                        for pr in prepared[(fi, condition, arm_b)]
                    ])
                    fold_deltas[f"{pair_name}/{condition}"].append(float(ap_a - ap_b))

        for key in bootstrap:
            bootstrap[key][b] = float(np.mean(fold_deltas[key]))

    results = {}
    for key, values in bootstrap.items():
        pair_name, condition = key.split("/", 1)
        results[key] = {
            "delta_mean": point_results[condition][pair_name]["delta_mean"],
            "per_fold_delta": point_results[condition][pair_name]["per_fold_delta"],
            "ci95_percentile": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "n_bootstrap": int(args.n_boot),
            "seed": int(args.seed),
        }

    output = {
        "title": "SECOM conditional bootstrap recomputation from archived predictions",
        "generated_date": "2026-11-08",
        "data_level": "saved outer-fold predictions; no raw feature data or retraining",
        "protocol": {
            "outer_folds": 5,
            "fold_unit": "outer test fold retained as a stratum",
            "within_fold_sampling": "positive and negative rows separately sampled with replacement, preserving fold-specific class counts",
            "paired_weights": "same sampled row-count weights used for C/B and all S1_30 evaluation-mask repeats within a fold",
            "S1_30_aggregation": "AP computed per evaluation-mask repeat, then averaged within fold",
            "estimand": "unweighted mean of five outer-fold AP differences",
            "bootstrap_replicates": int(args.n_boot),
            "random_seed": int(args.seed),
            "limitations": [
                "conditional on archived fitted models, training masks, and saved predictions",
                "does not include model retraining, training-sample resampling, or alpha reselection",
                "does not model dependence induced by overlapping outer training sets",
                "does not establish equivalence when a confidence interval includes zero",
            ],
        },
        "fold_partition": {
            "n_test_rows_total": len(test_row_ids),
            "n_unique_test_rows": len(set(test_row_ids)),
            "n_rows": len(test_row_ids),
            "fold_test_sizes": [int(len(f["test_rows"])) for f in fold_arrays],
            "fold_positive_counts": [int(np.sum(f["y_test"])) for f in fold_arrays],
        },
        "source_files": source_hashes,
        "archived_point_estimate_checks": archived_point_checks,
        "point_estimates": point_results,
        "bootstrap_results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "all_point_estimates_match_archived": all(
            x["absolute_difference"] < 1e-12 for x in archived_point_checks.values()
        ),
        "results": results,
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
