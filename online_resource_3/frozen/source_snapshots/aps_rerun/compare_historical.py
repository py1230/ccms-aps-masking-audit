"""Compare the completed APS rerun with archived probabilities and candidate scores.

No training, tuning, imputation of missing historical results, or significance
tests are performed. NPZ predictions take priority for point-estimate comparison;
the original formal candidate score table alone determines the replayed historical
selector, so candidates from different runs are not mixed during selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from verify_results import independent_ap, independent_selection

HERE = Path(__file__).resolve().parent
CONFIGS = ["M3", "C025", "B025", "C050", "B050", "C100", "B100", "A3", "A4"]
CONDITIONS = ["S0"] + [f"S1_{rate}_{s}" for rate in [10, 30] for s in [101, 202, 303]]
CONDITIONS += [f"S2_{s}" for s in [1101, 1202, 1303]] + [f"S3_{s}" for s in [2101, 2202, 2303]]
GROUPS = {name: [c for c in CONDITIONS if c == name or c.startswith(name + "_")]
          for name in ["S0", "S1_10", "S1_30", "S2", "S3"]}
FORMAL = "aps_formal_20261031T142351.json"
TESTSET = "aps_testset_20261031T145423.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nullable_mean(values):
    # Missing historical cells are not silently discarded to create a new average.
    return None if any(v is None for v in values) else float(np.mean(values))


def delta(new, old):
    return None if new is None or old is None else new - old


def compare(output_dir):
    output_dir = Path(output_dir)
    targets = ["new_vs_historical.csv", "fixed_and_selected_by_split.csv",
               "fixed_and_selected_summary.csv", "selection_comparison.json",
               "selection_candidates.csv", "comparison_summary.json"]
    for name in targets:
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg = read_json(HERE / "config.json")
    atol = cfg["historical_ap_comparison_tolerance"]
    run = HERE / "run_results"
    summary = read_json(run / "summary.json")
    assert summary["status"] == "complete" and summary["fits_completed"] == 125
    formal_path, test_path = HERE / "historical" / FORMAL, HERE / "historical" / TESTSET
    formal = {s["seed"]: s for s in read_json(formal_path)["splits"]}
    tests = {s["seed"]: s for s in read_json(test_path)["splits"]}
    source_hashes = {f"historical/{FORMAL}": sha(formal_path), f"historical/{TESTSET}": sha(test_path)}
    rows, choices, candidate_rows = [], [], []
    old_metrics, new_metrics = {}, {}
    max_new_metric_error = 0.0
    for split in cfg["split_seeds"]:
        dest = run / f"split{split}"
        metrics = read_json(dest / "metrics.json")
        selection = read_json(dest / "selection.json")
        current_path = dest / "ensemble_predictions.npz"
        current = np.load(current_path, allow_pickle=False)
        archives = {name: np.load(HERE / "historical" / f"{name}_seed{split}.npz", allow_pickle=False)
                    for name in ["aps", "arm3", "arm4"]}
        for name, archive in archives.items():
            assert np.array_equal(archive["train_rows"], current["train_rows"])
            assert np.array_equal(archive["val_rows"], current["val_rows"])
            for dataset in ["val", "test"]:
                assert np.array_equal(archive[f"y_{dataset}"], current[f"y_{dataset}"])
            source_hashes[f"historical/{name}_seed{split}.npz"] = sha(HERE / "historical" / f"{name}_seed{split}.npz")
        old_metrics[split], new_metrics[split] = {}, {}
        for config in CONFIGS:
            old_metrics[split][config], new_metrics[split][config] = {}, {}
            for dataset in ["val", "test"]:
                saved_json = formal[split] if dataset == "val" else tests[split]
                score_file = FORMAL if dataset == "val" else TESTSET
                archive_name = "arm3" if config == "A3" else "arm4" if config == "A4" else "aps"
                archive = archives[archive_name]
                for condition in CONDITIONS:
                    key = f"{config}__{dataset}__{condition}"
                    new_probability = current[key]
                    new_ap = independent_ap(current[f"y_{dataset}"], new_probability)
                    metric_error = abs(new_ap - metrics[config][f"{dataset}__{condition}"])
                    assert metric_error <= cfg["tolerances"]["ap_atol"]
                    max_new_metric_error = max(max_new_metric_error, metric_error)
                    old_prediction_available = key in archive.files
                    json_score = saved_json["metrics"].get(config, {}).get(condition, {}).get("ap")
                    old_ap, old_source, old_kind, max_probability_difference, alignment = None, "", "missing", None, ""
                    old_npz_vs_json_delta = None
                    if old_prediction_available:
                        old_probability = archive[key]
                        assert old_probability.shape == new_probability.shape
                        old_ap = independent_ap(archive[f"y_{dataset}"], old_probability)
                        old_source = f"historical/{archive_name}_seed{split}.npz::{key}"
                        old_kind = "recomputed_saved_prediction"
                        max_probability_difference = float(np.max(np.abs(new_probability - old_probability)))
                        alignment = "explicit_validation_row_indices_equal" if dataset == "val" else "archived_script_raw_test_order_and_equal_labels"
                        if json_score is not None:
                            old_npz_vs_json_delta = old_ap - float(json_score)
                    elif json_score is not None:
                        old_ap = float(json_score)
                        old_source = f"historical/{score_file}::split={split}/metrics/{config}/{condition}/ap"
                        old_kind = "archived_candidate_score_only"
                    difference = delta(new_ap, old_ap)
                    status = "missing_historical_reference" if old_ap is None else (
                        "numerically_consistent" if abs(difference) <= atol else "unexplained_difference")
                    rows.append({"split": split, "config": config, "dataset": dataset,
                                 "condition": condition, "old_ap": old_ap, "new_ap": new_ap,
                                 "new_minus_old_ap": difference,
                                 "numerical_match": None if difference is None else abs(difference) <= atol,
                                 "ap_tolerance": atol, "status": status,
                                 "old_prediction_available": old_prediction_available,
                                 "old_evidence_kind": old_kind, "old_source": old_source,
                                 "new_source": f"run_results/split{split}/ensemble_predictions.npz::{key}",
                                 "prediction_max_abs_difference": max_probability_difference,
                                 "prediction_row_alignment": alignment,
                                 "old_archived_json_ap": json_score,
                                 "old_npz_minus_json_ap": old_npz_vs_json_delta,
                                 "difference_cause": "not_established" if status == "unexplained_difference" else ""})
                    old_metrics[split][config][f"{dataset}__{condition}"] = old_ap
                    new_metrics[split][config][f"{dataset}__{condition}"] = new_ap
        # Replay from a single coherent old score table, including its old M3 base.
        old_selection_metrics = {name: {c: float(values[c]["ap"]) for c in CONDITIONS}
                                 for name, values in formal[split]["metrics"].items()
                                 if name in CONFIGS}
        new_selection_metrics = {name: {c: vals[f"val__{c}"] for c in CONDITIONS}
                                 for name, vals in new_metrics[split].items()}
        chosen = {"split": split, "historical_selection_source": f"historical/{FORMAL}",
                  "historical_replay_scope": "saved candidate scores, not retraining or prediction reconstruction for missing candidates",
                  "arms": {}}
        for arm in ["C", "B"]:
            old_choice, old_candidates = independent_selection(old_selection_metrics, arm, cfg["alphas"],
                                                               cfg["s0_tolerance"], cfg["tolerances"]["selection_tie_atol"])
            new_choice, new_candidates = independent_selection(new_selection_metrics, arm, cfg["alphas"],
                                                               cfg["s0_tolerance"], cfg["tolerances"]["selection_tie_atol"])
            assert new_choice["config"] == selection["chosen"][arm]["config"]
            assert new_choice["alpha"] == selection["chosen"][arm]["alpha"]
            chosen["arms"][arm] = {
                "historical_original_alpha": formal[split]["selection"][arm]["chosen_alpha"],
                "historical_replayed_complete_choice": old_choice,
                "new_complete_choice": new_choice,
                "selected_alpha_matches": old_choice["alpha"] == new_choice["alpha"]}
            for old, new in zip(old_candidates, new_candidates):
                assert old["alpha"] == new["alpha"]
                candidate_rows.append({"split": split, "arm": arm, "alpha": old["alpha"],
                                       "config": old["config"], "old_s0": old["s0"], "new_s0": new["s0"],
                                       "old_worst4": old["worst4"], "new_worst4": new["worst4"],
                                       "old_eligible": old["eligible"], "new_eligible": new["eligible"],
                                       "historical_score_source": f"historical/{FORMAL}",
                                       "new_prediction_source": f"run_results/split{split}/ensemble_predictions.npz"})
        choices.append(chosen)
        for archive in archives.values():
            archive.close()
        current.close()
    assert len(rows) == len(cfg["split_seeds"]) * len(CONFIGS) * 2 * len(CONDITIONS) == 1170
    write_csv(output_dir / "new_vs_historical.csv", rows)
    write_csv(output_dir / "selection_candidates.csv", candidate_rows)
    write_json(output_dir / "selection_comparison.json", choices)

    # Fixed estimates and selected-program estimates are explicitly distinct rows.
    split_summary = []
    for choice in choices:
        split = choice["split"]
        mappings = {"fixed_M3": ("M3", "M3"), "fixed_C025": ("C025", "C025"),
                    "fixed_B025": ("B025", "B025"), "fixed_A3": ("A3", "A3"),
                    "fixed_A4": ("A4", "A4")}
        for arm in ["C", "B"]:
            mappings[arm + "_selected"] = (choice["arms"][arm]["historical_replayed_complete_choice"]["config"],
                                          choice["arms"][arm]["new_complete_choice"]["config"])
        for dataset in ["val", "test"]:
            for group, conditions in GROUPS.items():
                values = {}
                for estimate, (old_config, new_config) in mappings.items():
                    old_values = [old_metrics[split][old_config][f"{dataset}__{c}"] for c in conditions]
                    new_values = [new_metrics[split][new_config][f"{dataset}__{c}"] for c in conditions]
                    old_value, new_value = nullable_mean(old_values), nullable_mean(new_values)
                    values[estimate] = (old_value, new_value)
                    split_summary.append({"split": split, "dataset": dataset, "condition_group": group,
                                          "estimate": estimate, "old_config": old_config, "new_config": new_config,
                                          "old_value": old_value, "new_value": new_value,
                                          "new_minus_old": delta(new_value, old_value),
                                          "old_complete": old_value is not None,
                                          "n_evaluation_inputs": len(conditions), "quantity": "AP"})
                for name, a, b in [("fixed_C_minus_B", "fixed_C025", "fixed_B025"),
                                   ("fixed_C_minus_M3", "fixed_C025", "fixed_M3"),
                                   ("fixed_B_minus_M3", "fixed_B025", "fixed_M3"),
                                   ("fixed_C_minus_A3", "fixed_C025", "fixed_A3"),
                                   ("fixed_A4_minus_B", "fixed_A4", "fixed_B025"),
                                   ("fixed_A3_minus_B", "fixed_A3", "fixed_B025"),
                                   ("selected_C_minus_B", "C_selected", "B_selected")]:
                    old_value = delta(values[a][0], values[b][0])
                    new_value = delta(values[a][1], values[b][1])
                    split_summary.append({"split": split, "dataset": dataset, "condition_group": group,
                                          "estimate": name, "old_config": f"{mappings[a][0]} - {mappings[b][0]}",
                                          "new_config": f"{mappings[a][1]} - {mappings[b][1]}",
                                          "old_value": old_value, "new_value": new_value,
                                          "new_minus_old": delta(new_value, old_value),
                                          "old_complete": old_value is not None,
                                          "n_evaluation_inputs": len(conditions), "quantity": "AP_difference"})
    write_csv(output_dir / "fixed_and_selected_by_split.csv", split_summary)
    mean_rows = []
    for dataset in ["val", "test"]:
        for group in GROUPS:
            for estimate in dict.fromkeys(r["estimate"] for r in split_summary):
                subset = [r for r in split_summary if (r["dataset"], r["condition_group"], r["estimate"]) == (dataset, group, estimate)]
                assert len(subset) == 5
                old_value = nullable_mean([r["old_value"] for r in subset])
                new_value = nullable_mean([r["new_value"] for r in subset])
                mean_rows.append({"dataset": dataset, "condition_group": group, "estimate": estimate,
                                  "old_mean": old_value, "new_mean": new_value,
                                  "new_minus_old": delta(new_value, old_value),
                                  "n_splits": len(subset), "historical_splits_available": sum(r["old_complete"] for r in subset),
                                  "aggregation": "equal split mean; per-pressure AP averaged within split",
                                  "inference": "descriptive; overlapping splits; no confidence interval"})
    write_csv(output_dir / "fixed_and_selected_summary.csv", mean_rows)
    counts = {status: sum(r["status"] == status for r in rows)
              for status in ["numerically_consistent", "unexplained_difference", "missing_historical_reference"]}
    by_config = {}
    for name in CONFIGS:
        subset = [r for r in rows if r["config"] == name]
        comparable = [r for r in subset if r["old_ap"] is not None]
        by_config[name] = {"cells": len(subset), "comparable": len(comparable),
                           "matching": sum(r["numerical_match"] is True for r in subset),
                           "max_abs_ap_difference": max(abs(r["new_minus_old_ap"]) for r in comparable),
                           "max_abs_probability_difference": max((r["prediction_max_abs_difference"] for r in subset
                                                                   if r["prediction_max_abs_difference"] is not None), default=None)}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "cells": len(rows),
              "status_counts": counts, "ap_tolerance": atol,
              "old_prediction_available_cells": sum(r["old_prediction_available"] for r in rows),
              "old_candidate_score_only_cells": sum(r["old_evidence_kind"] == "archived_candidate_score_only" for r in rows),
              "new_metrics_independent_ap_max_error": max_new_metric_error,
              "old_npz_json_max_ap_difference": max(abs(r["old_npz_minus_json_ap"]) for r in rows if r["old_npz_minus_json_ap"] is not None),
              "by_config": by_config, "historical_source_hashes": source_hashes,
              "selected_alpha_matches": sum(a["selected_alpha_matches"] for c in choices for a in c["arms"].values()),
              "selected_alpha_comparisons": 10,
              "headline": [r for r in mean_rows if r["condition_group"] in ["S0", "S1_30"]
                           and r["estimate"] in ["fixed_C_minus_B", "fixed_C_minus_A3", "selected_C_minus_B"]],
              "interpretation": ["Unexplained means no cause has been established by this numerical comparison.",
                                 "A version mismatch is a possible contributor, not an established explanation.",
                                 "Missing old results remain empty; means are withheld if constituent old cells are missing.",
                                 "The old complete selector is replayed from a single archived score table, including alpha=0.",
                                 "Both fixed-dose and selected-program comparisons are retrospective descriptive estimates."]}
    write_json(output_dir / "comparison_summary.json", result)
    print(json.dumps({"cells": len(rows), "status_counts": counts, "by_config": by_config}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    compare(args.output_dir)
