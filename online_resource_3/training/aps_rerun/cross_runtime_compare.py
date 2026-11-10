"""Controlled comparison of two APS runs, without loading cross-version models.

Scientific inputs and masks are compared directly. Each interpreter fingerprints
its own raw CSV parsing via the same archived aps_io source. Only NumPy arrays,
JSON and CSV are read across environments; no joblib file is deserialized here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHED = HERE / "environment_matched_run"
INITIAL_PYTHON = Path("/opt/anaconda3/bin/python")
MATCHED_PYTHON = Path("/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprint(value, kind):
    """Canonical little-endian C-order bytes; all float NaNs have one bit pattern."""
    if kind == "float64":
        canonical = np.array(value, dtype="<f8", order="C", copy=True)
        canonical.view("<u8")[np.isnan(canonical)] = np.uint64(0x7ff8000000000000)
    elif kind == "int64":
        canonical = np.array(value, dtype="<i8", order="C", copy=True)
    else:
        raise ValueError(kind)
    metadata = json.dumps({"shape": list(canonical.shape), "dtype": canonical.dtype.str,
                           "order": "C"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"shape": list(canonical.shape), "dtype": canonical.dtype.str,
            "sha256": hashlib.sha256(metadata + b"\0" + canonical.tobytes(order="C")).hexdigest()}


def fingerprint_raw(run_dir, output):
    # This mode is invoked separately under each interpreter. It uses the exact
    # source parser used by that run, not a parser native to the controlling process.
    run_dir = Path(run_dir).resolve()
    sys.path.insert(0, str(run_dir / "source"))
    from aps_io import load_aps_csv, split_features
    cfg = read_json(run_dir / "config.json")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(run_dir), "python": platform.python_version(),
              "executable": sys.executable, "numpy": np.__version__,
              "pandas": importlib.metadata.version("pandas"),
              "sklearn": importlib.metadata.version("scikit-learn"),
              "parser_sha256": sha(run_dir / "source" / "aps_io.py"),
              "canonicalization": "X little-endian float64 C-order with NaN bits 0x7ff8000000000000; y little-endian int64; columns UTF-8 compact JSON; shape/dtype included in array hashes",
              "datasets": {}}
    for dataset, key in [("train", "train_csv"), ("test", "test_csv")]:
        path = Path(cfg[key])
        frame, meta = load_aps_csv(path)
        x, y = split_features(frame)
        columns = [str(c) for c in frame.columns if c != "class"]
        serialized_columns = json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        result["datasets"][dataset] = {"raw_path": str(path), "raw_sha256": sha(path),
                                        "X": array_fingerprint(x, "float64"),
                                        "y": array_fingerprint(y, "int64"),
                                        "feature_names_sha256": hashlib.sha256(serialized_columns).hexdigest(),
                                        "feature_count": len(columns),
                                        "n_missing": int(np.isnan(x).sum()),
                                        "positive_count": int(y.sum()),
                                        "header_line_index": meta["header_line_index"]}
    write_json(output, result)
    print(json.dumps({"fingerprint_output": str(output), "python": result["python"],
                      "numpy": result["numpy"], "sklearn": result["sklearn"]}), flush=True)


def collect_fingerprints(initial_python, matched_python):
    specs = [(HERE, Path(initial_python), HERE / "audit" / "raw_matrix_fingerprint_initial.json"),
             (MATCHED, Path(matched_python), HERE / "audit" / "raw_matrix_fingerprint_matched.json")]
    results = []
    for run_dir, python, path in specs:
        if not path.exists():
            subprocess.run([str(python), str(Path(__file__).resolve()), "--fingerprint-one",
                            str(run_dir), "--fingerprint-output", str(path)], check=True)
        result = read_json(path)
        assert result["run_dir"] == str(run_dir.resolve())
        assert Path(result["executable"]).resolve() == python.resolve(), (result["executable"], str(python))
        assert result["parser_sha256"] == sha(run_dir / "source" / "aps_io.py")
        cfg = read_json(run_dir / "config.json")
        for dataset in ["train", "test"]:
            info = result["datasets"][dataset]
            assert info["raw_path"] == cfg[dataset + "_csv"]
            assert info["raw_sha256"] == sha(info["raw_path"])
        results.append(result)
    return results


def compare_arrays(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    exact = bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))
    error = float(np.max(np.abs(a - b))) if a.dtype.kind in "fci" and a.size else (0.0 if exact else None)
    return {"exact": exact, "max_abs_difference": error}


def compare_npz(path_a, path_b):
    records = {}
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        assert set(a.files) == set(b.files), (str(path_a), "key mismatch")
        for key in a.files:
            records[key] = compare_arrays(a[key], b[key])
    return records


def csv_records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return {(int(r["split"]), r["config"], r["dataset"], r["condition"]): r
                for r in csv.DictReader(f)}


def compare(initial_python, matched_python):
    out = HERE / "cross_runtime_comparison.json"
    csv_out = HERE / "environment_attribution.csv"
    if out.exists() or csv_out.exists():
        raise FileExistsError("Cross-runtime outputs exist; preserve them before another invocation")
    fingerprints = collect_fingerprints(initial_python, matched_python)
    cfg_a, cfg_b = [read_json(r / "config.json") for r in [HERE, MATCHED]]
    ignored_config_keys = ["design"]
    assert set(cfg_a) == set(cfg_b), "Configuration keys differ"
    configuration_checks = {key: cfg_a[key] == cfg_b[key] for key in cfg_a if key not in ignored_config_keys}
    source_files = {"run_aps.py", "historical/run_cb_firstlook.py"}
    source_a = {str(p.relative_to(HERE)) for p in (HERE / "source").glob("*.py")}
    source_b = {str(p.relative_to(MATCHED)) for p in (MATCHED / "source").glob("*.py")}
    assert source_a == source_b, "Source inventories differ"
    source_files |= source_a
    source_checks = {name: {"initial_sha256": sha(HERE / name), "matched_sha256": sha(MATCHED / name),
                            "identical": sha(HERE / name) == sha(MATCHED / name)}
                     for name in sorted(source_files)}
    index_checks = compare_npz(HERE / "split_indices.npz", MATCHED / "split_indices.npz")
    raw_comparisons = {}
    for dataset in ["train", "test"]:
        a, b = [f["datasets"][dataset] for f in fingerprints]
        raw_comparisons[dataset] = {key: a[key] == b[key] for key in
                                    ["raw_sha256", "X", "y", "feature_names_sha256", "feature_count", "n_missing", "positive_count", "header_line_index"]}
    run_a, run_b = HERE / "run_results", MATCHED / "run_results"
    for root in [run_a, run_b]:
        summary = read_json(root / "summary.json")
        assert summary["status"] == "complete" and summary["fits_completed"] == 125
    history_comparisons = [read_json(r / "comparison_summary.json") for r in [HERE, MATCHED]]
    assert history_comparisons[0]["historical_source_hashes"] == history_comparisons[1]["historical_source_hashes"]
    old_tables = [csv_records(r / "new_vs_historical.csv") for r in [HERE, MATCHED]]
    assert set(old_tables[0]) == set(old_tables[1]) and len(old_tables[0]) == 1170
    test_masks = compare_npz(run_a / "test_evaluation_masks.npz", run_b / "test_evaluation_masks.npz")
    counts = {"training_masks": 0, "training_masks_exact": 0,
              "evaluation_masks": len(test_masks), "evaluation_masks_exact": sum(x["exact"] for x in test_masks.values()),
              "model_prediction_arrays": 0, "model_prediction_arrays_exact": 0,
              "ensemble_prediction_arrays": 0, "ensemble_prediction_arrays_exact": 0,
              "ap_cells": 0, "ap_cells_exact": 0, "ap_cells_within_tolerance": 0,
              "historical_probability_arrays": 0, "initial_historical_probability_arrays_exact": 0,
              "matched_historical_probability_arrays_exact": 0,
              "historical_ap_cells": 0, "matched_historical_ap_cells_within_tolerance": 0}
    atol = cfg_a["tolerances"]["ap_atol"]
    max_errors = {"single_model_probability": 0.0, "ensemble_probability": 0.0,
                  "ap": 0.0, "matched_historical_probability": 0.0}
    split_records, attribution = [], []
    for split in cfg_a["split_seeds"]:
        a, b = run_a / f"split{split}", run_b / f"split{split}"
        preprocessing = compare_npz(a / "preprocessing.npz", b / "preprocessing.npz")
        masks = compare_npz(a / "evaluation_masks.npz", b / "evaluation_masks.npz")
        counts["evaluation_masks"] += len(masks)
        counts["evaluation_masks_exact"] += sum(x["exact"] for x in masks.values())
        training_inventory_a = {p.name for p in (a / "training_masks").glob("*.npz")}
        training_inventory_b = {p.name for p in (b / "training_masks").glob("*.npz")}
        assert training_inventory_a == training_inventory_b and len(training_inventory_a) == 25
        mask_checks = {}
        for name in sorted(training_inventory_a):
            check = compare_npz(a / "training_masks" / name, b / "training_masks" / name)
            assert set(check) == {"new_mask"}
            exact = check["new_mask"]["exact"]
            mask_checks[name] = exact
            counts["training_masks"] += 1
            counts["training_masks_exact"] += exact
        prediction_inventory_a = {p.name for p in (a / "predictions").glob("*.npz")}
        prediction_inventory_b = {p.name for p in (b / "predictions").glob("*.npz")}
        assert prediction_inventory_a == prediction_inventory_b and len(prediction_inventory_a) == 50
        single_model_differences = {}
        for name in sorted(prediction_inventory_a):
            check = compare_npz(a / "predictions" / name, b / "predictions" / name)
            assert check.pop("y")["exact"] and check.pop("row_indices")["exact"]
            assert len(check) == 13
            for result in check.values():
                counts["model_prediction_arrays"] += 1
                counts["model_prediction_arrays_exact"] += result["exact"]
                max_errors["single_model_probability"] = max(max_errors["single_model_probability"], result["max_abs_difference"])
            single_model_differences[name] = {"max_abs_difference": max(v["max_abs_difference"] for v in check.values()),
                                              "exact_arrays": sum(v["exact"] for v in check.values()), "arrays": len(check)}
        metrics_a, metrics_b = read_json(a / "metrics.json"), read_json(b / "metrics.json")
        assert set(metrics_a) == set(metrics_b)
        archives = {name: np.load(HERE / "historical" / f"{name}_seed{split}.npz", allow_pickle=False)
                    for name in ["aps", "arm3", "arm4"]}
        with np.load(a / "ensemble_predictions.npz", allow_pickle=False) as pa, np.load(b / "ensemble_predictions.npz", allow_pickle=False) as pb:
            assert set(pa.files) == set(pb.files)
            for key in ["train_rows", "val_rows", "test_rows", "y_val", "y_test"]:
                assert np.array_equal(pa[key], pb[key]), (split, key)
            for config in metrics_a:
                assert set(metrics_a[config]) == set(metrics_b[config])
                for cell, ap_a in metrics_a[config].items():
                    dataset, condition = cell.split("__")
                    key = f"{config}__{dataset}__{condition}"
                    pred_a, pred_b = pa[key], pb[key]
                    check = compare_arrays(pred_a, pred_b)
                    counts["ensemble_prediction_arrays"] += 1
                    counts["ensemble_prediction_arrays_exact"] += check["exact"]
                    max_errors["ensemble_probability"] = max(max_errors["ensemble_probability"], check["max_abs_difference"])
                    ap_b = metrics_b[config][cell]
                    ap_difference = ap_b - ap_a
                    counts["ap_cells"] += 1
                    counts["ap_cells_exact"] += ap_a == ap_b
                    counts["ap_cells_within_tolerance"] += abs(ap_difference) <= atol
                    max_errors["ap"] = max(max_errors["ap"], abs(ap_difference))
                    index = (split, config, dataset, condition)
                    table_a, table_b = old_tables[0][index], old_tables[1][index]
                    assert bool(table_a["old_ap"]) == bool(table_b["old_ap"])
                    if table_a["old_ap"]:
                        # Independent AP summation may differ at floating-point
                        # roundoff across NumPy/BLAS runtimes even for identical arrays.
                        assert abs(float(table_a["old_ap"]) - float(table_b["old_ap"])) <= atol
                    assert table_a["old_prediction_available"] == table_b["old_prediction_available"]
                    assert abs(float(table_a["new_ap"]) - ap_a) <= atol
                    assert abs(float(table_b["new_ap"]) - ap_b) <= atol
                    old_ap = float(table_b["old_ap"]) if table_b["old_ap"] else None
                    ap_recovered = old_ap is not None and abs(ap_b - old_ap) <= atol
                    if old_ap is not None:
                        counts["historical_ap_cells"] += 1
                        counts["matched_historical_ap_cells_within_tolerance"] += ap_recovered
                    old_available = table_b["old_prediction_available"].lower() == "true"
                    initial_old_error = matched_old_error = None
                    probability_recovered = None
                    if old_available:
                        archive_name = "arm3" if config == "A3" else "arm4" if config == "A4" else "aps"
                        old = archives[archive_name][key]
                        assert old.shape == pred_a.shape
                        initial_old_error = float(np.max(np.abs(pred_a - old)))
                        matched_old_error = float(np.max(np.abs(pred_b - old)))
                        probability_recovered = bool(np.array_equal(pred_b, old))
                        assert abs(float(table_a["prediction_max_abs_difference"]) - initial_old_error) <= 1e-15
                        assert abs(float(table_b["prediction_max_abs_difference"]) - matched_old_error) <= 1e-15
                        counts["historical_probability_arrays"] += 1
                        counts["initial_historical_probability_arrays_exact"] += bool(np.array_equal(pred_a, old))
                        counts["matched_historical_probability_arrays_exact"] += probability_recovered
                        max_errors["matched_historical_probability"] = max(max_errors["matched_historical_probability"], matched_old_error)
                    attribution.append({"split": split, "config": config, "dataset": dataset, "condition": condition,
                                        "historical_ap": old_ap, "initial_runtime_ap": ap_a, "matched_runtime_ap": ap_b,
                                        "matched_minus_initial_ap": ap_difference,
                                        "cross_runtime_probability_max_abs_difference": check["max_abs_difference"],
                                        "historical_prediction_available": old_available,
                                        "initial_vs_historical_probability_max_abs_difference": initial_old_error,
                                        "matched_vs_historical_probability_max_abs_difference": matched_old_error,
                                        "matched_historical_probabilities_exact": probability_recovered,
                                        "matched_historical_ap_within_tolerance": None if old_ap is None else ap_recovered,
                                        "initial_historical_ap_difference": None if old_ap is None else ap_a - old_ap,
                                        "matched_historical_ap_difference": None if old_ap is None else ap_b - old_ap,
                                        "attribution_scope": "joint_runtime_only; individual_library_cause_not_identified"})
        for archive in archives.values():
            archive.close()
        fit_a = {r["tag"]: r for r in read_json(a / "fit_records.json")}
        fit_b = {r["tag"]: r for r in read_json(b / "fit_records.json")}
        assert set(fit_a) == set(fit_b)
        split_records.append({"split": split, "preprocessing": preprocessing,
                              "evaluation_masks": masks, "training_masks_exact": mask_checks,
                              "single_model_predictions": single_model_differences,
                              "n_iter_comparison": {tag: {"initial": fit_a[tag]["n_iter"], "matched": fit_b[tag]["n_iter"]}
                                                    for tag in fit_a},
                              "initial_selection": read_json(a / "selection.json")["chosen"],
                              "matched_selection": read_json(b / "selection.json")["chosen"]})
        print(json.dumps({"event": "cross_runtime_split_complete", "split": split,
                          "counts": counts}), flush=True)
    assert counts["training_masks"] == 125 and counts["evaluation_masks"] == 78
    assert counts["model_prediction_arrays"] == 3250
    assert counts["ensemble_prediction_arrays"] == counts["ap_cells"] == len(attribution) == 1170
    assert counts["historical_probability_arrays"] == 650
    identical_inputs = (all(configuration_checks.values())
                        and all(s["identical"] for s in source_checks.values())
                        and all(v["exact"] for v in index_checks.values())
                        and all(all(d.values()) for d in raw_comparisons.values())
                        and all(all(v["exact"] for v in s["preprocessing"].values()) for s in split_records)
                        and counts["training_masks_exact"] == 125 and counts["evaluation_masks_exact"] == 78)
    for row in attribution:
        row["scientific_inputs_identical"] = identical_inputs
        if not identical_inputs:
            row["attribution_scope"] = "unresolved_input_difference; runtime_attribution_not_supported"
    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(attribution[0]))
        writer.writeheader()
        writer.writerows(attribution)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "completed", "scientific_inputs_identical": identical_inputs,
              "configuration_checks": configuration_checks,
              "non_scientific_config_fields_excluded": ignored_config_keys,
              "source_checks": source_checks, "split_index_checks": index_checks,
              "raw_parser_fingerprints": fingerprints, "raw_parser_comparison": raw_comparisons,
              "runtime_environment": {"initial": read_json(run_a / "environment.json"),
                                      "matched": read_json(run_b / "environment.json")},
              "counts": counts, "max_errors": max_errors, "splits": split_records,
              "historical_comparison_status_counts": {"initial": history_comparisons[0]["status_counts"],
                                                       "matched": history_comparisons[1]["status_counts"]},
              "history_probability_restored_exactly": counts["matched_historical_probability_arrays_exact"] == 650,
              "interpretation": [
                  "When scientific_inputs_identical is true, observed differences are bounded to the jointly changed software/runtime environment under these fixed data, indices, preprocessing and masks.",
                  "Python, NumPy, pandas, scikit-learn and their dependencies changed together; this does not identify the causal contribution of any one library.",
                  "A single run per runtime does not characterize the distribution of runtime effects or establish general portability guarantees.",
                  "Exact recovery of saved probabilities is current reproduction evidence, not proof that historical training occurred as claimed.",
                  "Candidate scores without old probabilities support only AP comparison; missing old candidate results remain missing.",
                  "No joblib files were loaded across versions by this comparison script."]}
    write_json(out, result)
    print(json.dumps({"scientific_inputs_identical": identical_inputs, "counts": counts,
                      "max_errors": max_errors}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-one", type=Path)
    parser.add_argument("--fingerprint-output", type=Path)
    parser.add_argument("--fingerprints-only", action="store_true")
    parser.add_argument("--initial-python", type=Path, default=INITIAL_PYTHON)
    parser.add_argument("--matched-python", type=Path, default=MATCHED_PYTHON)
    args = parser.parse_args()
    if args.fingerprint_one:
        if args.fingerprint_output is None:
            parser.error("--fingerprint-output is required with --fingerprint-one")
        fingerprint_raw(args.fingerprint_one, args.fingerprint_output)
    elif args.fingerprints_only:
        records = collect_fingerprints(args.initial_python, args.matched_python)
        print(json.dumps({"status": "fingerprints_collected", "runtimes": [r["executable"] for r in records]}))
    else:
        compare(args.initial_python, args.matched_python)
