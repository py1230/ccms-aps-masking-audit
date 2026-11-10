"""Independent numerical and artifact checks for the 2026-11-08 APS rerun.

No fitting is performed.  CSV parsing, masks, median preprocessing, AP (including
ties), and selection are implemented here independently of the training runner.
Model reloading verifies saved predictions but is not a second training run.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def load_raw(path):
    """Parse raw APS CSV without relying on source/aps_io.py."""
    with Path(path).open(encoding="utf-8") as f:
        for header, line in enumerate(f):
            fields = [s.strip().strip("\"'").lower() for s in line.rstrip().split(",")]
            if "class" in fields:
                break
        else:
            raise ValueError("No class header in raw APS CSV")
    frame = pd.read_csv(path, skiprows=header, na_values=["na", "NA", "Na", "NaN", "nan", "?", ""],
                        low_memory=False)
    frame.columns = frame.columns.str.strip()
    labels = frame.pop("class").str.lower()
    assert labels.isin(["pos", "neg"]).all()
    features = frame.to_numpy(dtype=np.float64)
    assert np.isfinite(features[~np.isnan(features)]).all()
    return features, (labels.to_numpy() == "pos").astype(np.int64), list(frame.columns)


def independent_ap(labels, scores):
    """Noninterpolated average precision; all observations tied at a score enter together."""
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=np.float64)
    assert y.ndim == p.ndim == 1 and y.shape == p.shape and len(y) > 0
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
    positives = int(y.sum())
    assert 0 < positives < len(y)
    order = np.argsort(-p, kind="stable")
    s, t = p[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    true_positives = np.cumsum(t, dtype=np.int64)[ends]
    precision = true_positives / (ends + 1)
    newly_recalled = np.diff(np.r_[0, true_positives]) / positives
    return float(np.dot(newly_recalled, precision))


def median_fit(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def model_matrix(x, medians):
    missing = np.isnan(x)
    values = np.where(missing, medians[None, :], x)
    return np.concatenate((values, missing.astype(np.float64)), axis=1)


def training_mask(x, labels, arm, alpha, seed):
    """Independent full-matrix RNG reconstruction of C/B and exact-count A3/A4."""
    observed = ~np.isnan(x)
    counts = np.bincount(labels, minlength=2)
    q = np.stack([(~observed[labels == c]).mean(axis=0) for c in [0, 1]])
    maximum = q.max(axis=0)
    qbar = (q * (counts / len(labels))[:, None]).sum(axis=0)
    expected = len(labels) * alpha * (maximum - qbar)
    if arm in ("M3", "shared_zero") or alpha == 0:
        return np.zeros(x.shape, dtype=bool), expected
    reference = {"A3": "C", "A4": "B"}.get(arm, arm)
    if reference == "C":
        probabilities = np.zeros_like(q)
        np.divide(alpha * (maximum[None, :] - q), 1.0 - q,
                  out=probabilities, where=q < 1)
        probabilities = np.clip(probabilities, 0, 1)[labels]
    elif reference == "B":
        probabilities = np.zeros(x.shape[1])
        np.divide(alpha * (maximum - qbar), 1.0 - qbar,
                  out=probabilities, where=qbar < 1)
        probabilities = np.clip(probabilities, 0, 1)[None, :]
    else:
        raise ValueError(arm)
    mask = observed & (np.random.default_rng(seed).random(x.shape) < probabilities)
    if arm in ("A3", "A4"):
        target_counts = mask.sum(axis=0)
        random = np.random.default_rng(seed * 1000 + (7 if arm == "A3" else 11))
        mask = np.zeros(x.shape, dtype=bool)
        for j, count in enumerate(target_counts):
            if count:
                visible_rows = np.flatnonzero(observed[:, j])
                assert count <= len(visible_rows)
                mask[random.choice(visible_rows, size=int(count), replace=False), j] = True
        assert np.array_equal(mask.sum(axis=0), target_counts)
    return mask, expected


def evaluation_mask(x, condition):
    visible = ~np.isnan(x)
    if condition == "S0":
        return np.zeros(x.shape, dtype=bool)
    terms = condition.split("_")
    rng = np.random.default_rng(int(terms[-1]))
    if terms[0] == "S1":
        return visible & (rng.random(x.shape) < int(terms[1]) / 100)
    mask = np.zeros(x.shape, dtype=bool)
    if terms[0] == "S2":
        rows = rng.choice(len(x), size=int(round(0.5 * len(x))), replace=False)
        mask[rows] = visible[rows] & (rng.random((len(rows), x.shape[1])) < 0.6)
    elif terms[0] == "S3":
        columns = rng.choice(x.shape[1], size=max(1, int(round(0.3 * x.shape[1]))), replace=False)
        mask[:, columns] = visible[:, columns]
    else:
        raise ValueError(condition)
    return mask


def independent_selection(metrics, arm, alphas, tolerance, tie_tolerance):
    conditions = ["S0", "S1_10_101", "S1_10_202", "S1_10_303"]
    rows = []
    base = metrics["M3"]["S0"]
    for alpha in alphas:
        name = "M3" if alpha == 0 else f"{arm}{round(alpha * 100):03d}"
        scores = metrics[name]
        rows.append({"alpha": alpha, "config": name, "s0": scores["S0"],
                     "worst4": min(scores[k] for k in conditions),
                     "eligible": bool(scores["S0"] >= base - tolerance)})
    feasible = [r for r in rows if r["eligible"]]
    best = max(r["worst4"] for r in feasible)
    winner = min([r for r in feasible if best - r["worst4"] <= tie_tolerance],
                 key=lambda r: r["alpha"])
    return winner, rows


def check_ap_examples():
    # Manually derived values test correct handling of tied thresholds.
    cases = [([1, 0], [0.5, 0.5], 0.5),
             ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], 5 / 6),
             ([0, 1, 1, 0], [0.9, 0.8, 0.8, 0.7], 2 / 3),
             ([1, 0, 1, 0], [1, 1, 1, 1], 0.5)]
    for labels, scores, expected in cases:
        assert abs(independent_ap(labels, scores) - expected) < 1e-14
    return len(cases)


def relative_path(path):
    path = Path(path)
    return path if path.is_absolute() else HERE / path


def allclose(actual, expected, atol, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert error <= atol, (label, error, atol)
    return error


def expected_models(config):
    yield "M3", "M3", 0.0, None
    for alpha in config["alphas"]:
        if alpha == 0:
            continue
        for arm in ["C", "B"]:
            for seed in config["train_mask_seeds"]:
                yield f"{arm}{round(alpha * 100):03d}_seed{seed}", arm, alpha, seed
    for arm in ["A3", "A4"]:
        for seed in config["train_mask_seeds"]:
            yield f"{arm}_seed{seed}", arm, 0.25, seed


def verify(output):
    started = time.perf_counter()
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Independent CSV parsing, mask generation, median calculation, tied-threshold AP and selector; saved sklearn models reloaded without fitting.",
              "limitations": ["Model reload is not independent retraining.",
                              "Current artifact integrity is not evidence of historical training provenance.",
                              "Repeated splits are not independent datasets; no new confidence intervals are computed."],
              "checked": {"freeze_files": 0, "models": 0, "training_masks": 0,
                          "evaluation_masks": 0, "prediction_arrays": 0,
                          "ensemble_arrays": 0, "ap_cells": 0, "selections": 0,
                          "count_matched_masks": 0, "diagnostic_files": 0},
              "max_errors": {"prediction": 0.0, "ensemble": 0.0, "ap": 0.0,
                             "expected_budget": 0.0, "selection_score": 0.0},
              "splits": []}
    try:
        report["ap_manual_examples"] = check_ap_examples()
        cfg = read_json(HERE / "config.json")
        freeze = read_json(HERE / "freeze.json")
        report["config_sha256"] = sha256(HERE / "config.json")
        report["freeze_sha256"] = sha256(HERE / "freeze.json")
        for name, entry in freeze["files"].items():
            path = relative_path(entry.get("path", name))
            assert path.is_file(), str(path)
            assert sha256(path) == entry["sha256"], ("Freeze hash mismatch", name)
            report["checked"]["freeze_files"] += 1
        assert report["checked"]["freeze_files"] > 0
        tolerances = cfg["tolerances"]
        pred_atol, ap_atol = tolerances["prediction_atol"], tolerances["ap_atol"]
        budget_atol = tolerances["budget_atol"]
        x_all, y_all, feature_names = load_raw(relative_path(cfg["train_csv"]))
        x_test, y_test, test_features = load_raw(relative_path(cfg["test_csv"]))
        assert x_all.shape == (60000, 170) and x_test.shape == (16000, 170)
        assert y_all.sum() == 1000 and y_test.sum() == 375
        assert feature_names == test_features
        raw_files = [relative_path(cfg[k]) for k in ["train_csv", "test_csv"]]
        frozen_paths = {relative_path(e.get("path", n)).resolve() for n, e in freeze["files"].items()}
        assert all(p.resolve() in frozen_paths for p in raw_files), "Both raw CSVs must be frozen"
        report["raw_csv_sha256"] = {p.name: sha256(p) for p in raw_files}
        indices = np.load(HERE / "split_indices.npz", allow_pickle=False)
        run = HERE / "run_results"
        conditions = ["S0"] + [f"S1_{rate}_{s}" for rate in [10, 30] for s in [101, 202, 303]]
        conditions += [f"S2_{s}" for s in [1101, 1202, 1303]] + [f"S3_{s}" for s in [2101, 2202, 2303]]
        models = list(expected_models(cfg))
        assert len(models) == 25
        # Only generated masks enter evaluation features; labels are never inputs to this generator.
        test_masks = {}
        with np.load(run / "test_evaluation_masks.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(conditions)
            for condition in conditions:
                mask = evaluation_mask(x_test, condition)
                assert np.array_equal(saved[condition], mask), ("test evaluation mask", condition)
                test_masks[condition] = mask
                report["checked"]["evaluation_masks"] += 1
        for split in cfg["split_seeds"]:
            destination = run / f"split{split}"
            train_rows, val_rows, reserve_rows = [indices[f"s{split}_{k}"] for k in ["train", "val", "reserve"]]
            for rows, size, positives in [(train_rows, 42000, 700), (val_rows, 9000, 150), (reserve_rows, 9000, 150)]:
                assert rows.ndim == 1 and len(rows) == size and np.issubdtype(rows.dtype, np.integer)
                assert len(np.unique(rows)) == size and rows.min() >= 0 and rows.max() < len(y_all)
                assert int(y_all[rows].sum()) == positives
            combined = np.concatenate([train_rows, val_rows, reserve_rows])
            assert np.array_equal(np.sort(combined), np.arange(len(y_all))), "Split is not a partition"
            with np.load(HERE / "historical" / f"aps_seed{split}.npz", allow_pickle=False) as history:
                assert np.array_equal(train_rows, history["train_rows"])
                assert np.array_equal(val_rows, history["val_rows"])
                assert np.array_equal(y_all[val_rows], history["y_val"])
                assert np.array_equal(y_test, history["y_test"])
            x, yy = x_all[train_rows], y_all[train_rows]
            medians = median_fit(x)
            miss = np.isnan(x)
            counts = np.bincount(yy, minlength=2)
            class_missing = np.stack([miss[yy == c].sum(axis=0) for c in [0, 1]])
            q = class_missing / counts[:, None]
            with np.load(destination / "preprocessing.npz", allow_pickle=False) as pp:
                assert np.array_equal(pp["train_rows"], train_rows)
                assert list(pp["feature_names"].astype(str)) == feature_names
                allclose(pp["medians"], medians, 0.0, "medians")
                allclose(pp["q"], q, 0.0, "class missing rates")
                assert np.array_equal(pp["class_counts"], counts)
                assert np.array_equal(pp["missing_counts"], class_missing)
            rowsets = {"val": val_rows, "test": np.arange(len(y_test))}
            ys = {"val": y_all[val_rows], "test": y_test}
            raw_eval = {"val": x_all[val_rows], "test": x_test}
            matrices = {"val": {}, "test": {}}
            with np.load(destination / "evaluation_masks.npz", allow_pickle=False) as saved:
                assert set(saved.files) == set(conditions)
                for condition in conditions:
                    mask = evaluation_mask(raw_eval["val"], condition)
                    assert np.array_equal(saved[condition], mask), (split, "val evaluation mask", condition)
                    report["checked"]["evaluation_masks"] += 1
                    for dataset, use_mask in [("val", mask), ("test", test_masks[condition])]:
                        xm = raw_eval[dataset].copy()
                        xm[use_mask] = np.nan
                        matrices[dataset][condition] = model_matrix(xm, medians)
            model_files = {p.stem for p in (destination / "models").glob("*.joblib")}
            expected_tags = {m[0] for m in models}
            assert model_files == expected_tags, (split, "model inventory", model_files ^ expected_tags)
            ensemble_components = {}
            fixed_reference_counts = {}
            iteration_counts = {}
            for tag, arm, alpha, seed in models:
                # No invocation of source/ccms.py or the runner's mask generator.
                mask, expected_budget = training_mask(x, yy, arm, alpha, seed)
                assert not (mask & miss).any()
                with np.load(destination / "training_masks" / f"{tag}.npz", allow_pickle=False) as saved:
                    assert np.array_equal(saved["new_mask"], mask), (split, tag, "training mask mismatch")
                report["checked"]["training_masks"] += 1
                column_counts = mask.sum(axis=0)
                if arm in ["C", "B"] and alpha == 0.25:
                    fixed_reference_counts[(arm, seed)] = column_counts
                if arm in ["A3", "A4"]:
                    reference = fixed_reference_counts[("C" if arm == "A3" else "B", seed)]
                    assert np.array_equal(column_counts, reference), (split, tag, "count matching")
                    report["checked"]["count_matched_masks"] += 1
                class_counts = np.stack([mask[yy == c].sum(axis=0) for c in [0, 1]])
                q_after = (class_missing + class_counts) / counts[:, None]
                with np.load(destination / "diagnostics" / f"{tag}.npz", allow_pickle=False) as diag:
                    assert np.array_equal(diag["row_deleted"], mask.sum(axis=1))
                    assert np.array_equal(diag["col_deleted"], column_counts)
                    assert np.array_equal(diag["class_col_deleted"], class_counts)
                    allclose(diag["q_after"], q_after, 1e-15, "postmask class missing rates")
                    # This field is the unconditional budget of the paired C/B
                    # mechanism; A3/A4 conditional counts are recorded separately.
                    err = allclose(diag["expected_col"], expected_budget, budget_atol, "expected budget")
                    assert np.array_equal(diag["reference_col_counts"], column_counts)
                    report["max_errors"]["expected_budget"] = max(report["max_errors"]["expected_budget"], err)
                report["checked"]["diagnostic_files"] += 1
                payload = joblib.load(destination / "models" / f"{tag}.joblib")
                assert payload["arm"] in (["M3", "shared_zero"] if arm == "M3" else [arm])
                assert payload["alpha"] == alpha and payload["mask_seed"] == seed
                assert payload["model_seed"] == cfg["model_params"]["random_state"]
                assert np.array_equal(payload["train_rows"], train_rows)
                allclose(payload["medians"], medians, 0.0, "model preprocessing")
                model = payload["model"]
                assert np.array_equal(model.classes_, [0, 1]) and model.n_features_in_ == 340
                assert model.get_params() == cfg["model_params"], (split, tag, "model parameters")
                iteration_counts[tag] = int(model.n_iter_)
                config_name = tag.split("_seed")[0]
                for dataset in ["val", "test"]:
                    with np.load(destination / "predictions" / f"{tag}__{dataset}.npz", allow_pickle=False) as pred:
                        assert set(pred.files) == {"row_indices", "y", *conditions}
                        assert np.array_equal(pred["row_indices"], rowsets[dataset])
                        assert np.array_equal(pred["y"], ys[dataset])
                        for condition in conditions:
                            probability = pred[condition]
                            assert probability.shape == ys[dataset].shape
                            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
                            loaded_probability = model.predict_proba(matrices[dataset][condition])[:, 1]
                            err = allclose(loaded_probability, probability, pred_atol,
                                           f"{split}/{tag}/{dataset}/{condition}")
                            report["max_errors"]["prediction"] = max(report["max_errors"]["prediction"], err)
                            report["checked"]["prediction_arrays"] += 1
                            key = f"{config_name}__{dataset}__{condition}"
                            ensemble_components.setdefault(key, []).append(probability.copy())
                report["checked"]["models"] += 1
                del model, payload, mask
            metrics = read_json(destination / "metrics.json")
            computed_metrics = {}
            with np.load(destination / "ensemble_predictions.npz", allow_pickle=False) as ensembles:
                assert np.array_equal(ensembles["y_val"], ys["val"])
                assert np.array_equal(ensembles["y_test"], ys["test"])
                assert np.array_equal(ensembles["val_rows"], val_rows)
                assert np.array_equal(ensembles["test_rows"], rowsets["test"])
                assert np.array_equal(ensembles["train_rows"], train_rows)
                assert set(ensembles.files) == set(ensemble_components) | {"y_val", "y_test", "val_rows", "test_rows", "train_rows"}
                for key, components in ensemble_components.items():
                    config_name, dataset, condition = key.split("__")
                    assert len(components) == (1 if config_name == "M3" else 3)
                    rebuilt = np.mean(components, axis=0)
                    err = allclose(ensembles[key], rebuilt, pred_atol, "ensemble")
                    report["max_errors"]["ensemble"] = max(report["max_errors"]["ensemble"], err)
                    report["checked"]["ensemble_arrays"] += 1
                    ap = independent_ap(ys[dataset], ensembles[key])
                    err = allclose(metrics[config_name][f"{dataset}__{condition}"], ap, ap_atol, "AP")
                    report["max_errors"]["ap"] = max(report["max_errors"]["ap"], err)
                    report["checked"]["ap_cells"] += 1
                    computed_metrics.setdefault(config_name, {})[f"{dataset}__{condition}"] = ap
            assert set(metrics) == set(computed_metrics)
            for name in computed_metrics:
                assert set(metrics[name]) == set(computed_metrics[name])
            validation_metrics = {name: {condition: values[f"val__{condition}"] for condition in conditions}
                                  for name, values in computed_metrics.items()}
            selection = read_json(destination / "selection.json")
            assert selection["selected_before_test_prediction"] is True
            choices = {}
            for arm in ["C", "B"]:
                chosen, candidates = independent_selection(validation_metrics, arm, cfg["alphas"],
                                                          cfg["s0_tolerance"], tolerances["selection_tie_atol"])
                assert selection["chosen"][arm]["alpha"] == chosen["alpha"]
                assert selection["chosen"][arm]["config"] == chosen["config"]
                saved_candidates = selection["candidates"][arm]
                assert len(saved_candidates) == len(candidates) == 4
                for saved, fresh in zip(saved_candidates, candidates):
                    for key in ["alpha", "config", "eligible"]:
                        assert saved[key] == fresh[key], (split, arm, key)
                    for key in ["s0", "worst4"]:
                        err = allclose(saved[key], fresh[key], ap_atol, "selection score")
                        report["max_errors"]["selection_score"] = max(report["max_errors"]["selection_score"], err)
                choices[arm] = chosen["alpha"]
                report["checked"]["selections"] += 1
            report["splits"].append({"split": split, "status": "passed", "selected_alphas": choices,
                                     "n_train": len(train_rows), "n_val": len(val_rows),
                                     "n_test": len(y_test), "models": len(models),
                                     "n_iter_by_model": iteration_counts})
            print(json.dumps({"event": "verification_split_complete", "split": split,
                              "models_checked": report["checked"]["models"],
                              "max_errors": report["max_errors"]}), flush=True)
            del matrices, ensemble_components, raw_eval, x
        assert report["checked"]["models"] == 125
        assert report["checked"]["training_masks"] == 125
        assert report["checked"]["prediction_arrays"] == 3250
        assert report["checked"]["ap_cells"] == 1170
        assert report["checked"]["selections"] == 10
        assert report["checked"]["count_matched_masks"] == 30
        assert report["checked"]["evaluation_masks"] == 78
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter() - started
        write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"manual_AP_examples": check_ap_examples(), "status": "passed"}))
    else:
        if args.output.exists():
            raise SystemExit(f"Refusing to overwrite {args.output}; use --output with a new filename")
        with threadpool_limits(limits=2):
            result = verify(args.output)
        print(json.dumps({"status": result["status"], "checked": result["checked"],
                          "max_errors": result["max_errors"]}, indent=2))
