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


def model_matrix(x, medians, channel):
    missing = np.isnan(x)
    if channel == "MO":
        return missing.astype(np.float64)
    values = np.where(missing, medians[None, :], x)
    if channel == "M2":
        return values
    assert channel == "M3"
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
    base = metrics["BASE"]["S0"]
    for alpha in alphas:
        name = "BASE" if alpha == 0 else f"{arm}{round(alpha * 100):03d}"
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


def expected_models(config, repeat):
    yield 'BASE', 'BASE', 0.0, None
    for alpha in config['alphas']:
        if alpha == 0:
            continue
        for arm in ['C', 'B']:
            for seed in repeat['mask_seeds']:
                yield f'{arm}{round(alpha * 100):03d}_seed{seed}', arm, alpha, seed


def check_channel_examples():
    x = np.array([[1., np.nan], [np.nan, 4.], [3., 6.]])
    med = np.array([2., 5.])
    values = np.array([[1., 5.], [2., 4.], [3., 6.]])
    flags = np.array([[0., 1.], [1., 0.], [0., 0.]])
    assert np.array_equal(model_matrix(x, med, 'M2'), values)
    assert np.array_equal(model_matrix(x, med, 'MO'), flags)
    assert np.array_equal(model_matrix(x, med, 'M3'), np.column_stack([values, flags]))
    return 3


def verify(output, selected_channel=None):
    started = time.perf_counter()
    report = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha256(__file__),
        'method': 'Separate CSV parsing, median calculation, channel construction, mask RNG, tied-threshold AP and selector; saved model reloading without fitting.',
        'limitations': ['Reloading is not independent retraining.',
                       'Fixed rows and pressure inputs; only joint conditional training randomness repeated.',
                       'Overlapping splits are not independent datasets; three groups do not establish statistical stability.',
                       'Verification is performed by another assistant in the same session, not blind or cross-model peer review.'],
        'checked': {'freeze_files': 0, 'models': 0, 'training_masks': 0,
                    'evaluation_masks': 0, 'prediction_arrays': 0, 'ensemble_arrays': 0,
                    'ap_cells': 0, 'selections': 0, 'diagnostic_files': 0, 'summary_cells': 0},
        'max_errors': {'prediction': 0., 'ensemble': 0., 'ap': 0., 'expected_budget': 0.,
                       'selection_score': 0., 'summary': 0.}, 'cells': [],
        'pairing_fingerprints': {},
    }
    try:
        report['ap_manual_examples'] = check_ap_examples()
        report['channel_manual_examples'] = check_channel_examples()
        cfg = read_json(HERE / 'config.json')
        fr = read_json(HERE / 'freeze.json')
        report['config_sha256'] = sha256(HERE / 'config.json')
        report['freeze_sha256'] = sha256(HERE / 'freeze.json')
        report['runner_sha256'] = sha256(HERE / 'run_channels.py')
        for name, row in fr['files'].items():
            p = relative_path(row.get('path', name))
            assert p.is_file() and sha256(p) == row['sha256'], ('Freeze mismatch', name)
            report['checked']['freeze_files'] += 1
        assert report['checked']['freeze_files'] > 0
        frozen_paths = {relative_path(row.get('path', name)).resolve() for name, row in fr['files'].items()}
        assert all(relative_path(cfg[k]).resolve() in frozen_paths for k in ['train_csv', 'test_csv'])
        assert (HERE / 'split_indices.npz').resolve() in frozen_paths
        assert (HERE / 'config.json').resolve() in frozen_paths and (HERE / 'run_channels.py').resolve() in frozen_paths
        tol = cfg['tolerances']; pred_atol, ap_atol = tol['prediction_atol'], tol['ap_atol']
        x_all, y_all, names = load_raw(relative_path(cfg['train_csv']))
        x_test, y_test, test_names = load_raw(relative_path(cfg['test_csv']))
        assert x_all.shape == (60000, 170) and x_test.shape == (16000, 170)
        assert y_all.sum() == 1000 and y_test.sum() == 375 and names == test_names
        assert sha256(HERE / 'split_indices.npz') == cfg['split_source_sha256']
        with np.load(HERE / 'split_indices.npz', allow_pickle=False) as z:
            indices = {k: z[k] for k in z.files}
        conditions = ['S0'] + [f'S1_{r}_{s}' for r in [10, 30] for s in [101, 202, 303]]
        conditions += [f'S2_{s}' for s in [1101, 1202, 1303]] + [f'S3_{s}' for s in [2101, 2202, 2303]]
        groups = ['S0', 'S1_10', 'S1_30', 'S2', 'S3']
        assert cfg['selection_conditions'] == ['S0', 'S1_10_101', 'S1_10_202', 'S1_10_303']
        channels = [selected_channel] if selected_channel else cfg['channels']
        assert set(channels).issubset({'M2', 'M3', 'MO'})
        report['channels'] = channels
        test_masks = {c: evaluation_mask(x_test, c) for c in conditions}
        pairing = {}
        for channel in channels:
            run = HERE / 'run_results' / channel
            env = read_json(run / 'environment.json')
            for k, expected in cfg['input_environment'].items():
                assert env[k] == expected, (channel, 'environment', k)
            summary = read_json(run / 'summary.json')
            assert summary['status'] == 'complete' and summary['fits_completed'] == 285
            assert len(summary['cells']) == 15
            summary_map = {(r['repeat'], r['split']): r for r in summary['cells']}
            assert len(summary_map) == 15
            with np.load(run / 'test_evaluation_masks.npz', allow_pickle=False) as z:
                assert set(z.files) == set(conditions)
                for cond in conditions:
                    assert np.array_equal(z[cond], test_masks[cond]), (channel, 'test mask', cond)
                    report['checked']['evaluation_masks'] += 1
            for rep in cfg['repeats']:
                models = list(expected_models(cfg, rep)); assert len(models) == 19
                params = {**cfg['model_params'], 'random_state': rep['model_seed']}
                for split in cfg['split_seeds']:
                    dest = run / f'rep{rep["id"]}' / f'split{split}'
                    tr, va, rs = [indices[f's{split}_{k}'] for k in ['train', 'val', 'reserve']]
                    for rows, size, pos in [(tr, 42000, 700), (va, 9000, 150), (rs, 9000, 150)]:
                        assert len(rows) == size and np.issubdtype(rows.dtype, np.integer)
                        assert len(np.unique(rows)) == size and rows.min() >= 0 and rows.max() < len(y_all)
                        assert int(y_all[rows].sum()) == pos
                    assert np.array_equal(np.sort(np.concatenate([tr, va, rs])), np.arange(len(y_all)))
                    x, yy = x_all[tr], y_all[tr]
                    med = median_fit(x); missing = np.isnan(x)
                    counts = np.bincount(yy, minlength=2)
                    class_missing = np.stack([missing[yy == c].sum(0) for c in [0, 1]])
                    q = class_missing / counts[:, None]
                    with np.load(dest / 'preprocessing.npz', allow_pickle=False) as pp:
                        assert np.array_equal(pp['train_rows'], tr) and list(pp['feature_names']) == names
                        allclose(pp['medians'], med, 0., 'medians')
                        allclose(pp['q'], q, 0., 'q')
                        assert np.array_equal(pp['class_counts'], counts) and np.array_equal(pp['missing_counts'], class_missing)
                    ys = {'val': y_all[va], 'test': y_test}
                    rowsets = {'val': va, 'test': np.arange(len(y_test))}
                    raw = {'val': x_all[va], 'test': x_test}
                    matrices = {'val': {}, 'test': {}}
                    cell_fingerprint = hashlib.sha256()
                    for arr in [tr, va, rs, med]:
                        cell_fingerprint.update(arr.tobytes())
                    with np.load(dest / 'evaluation_masks.npz', allow_pickle=False) as z:
                        assert set(z.files) == set(conditions)
                        for cond in conditions:
                            val_mask = evaluation_mask(raw['val'], cond)
                            assert np.array_equal(z[cond], val_mask), (channel, rep['id'], split, cond)
                            report['checked']['evaluation_masks'] += 1
                            cell_fingerprint.update(val_mask.tobytes()); cell_fingerprint.update(test_masks[cond].tobytes())
                            for ds, mask in [('val', val_mask), ('test', test_masks[cond])]:
                                xm = raw[ds].copy(); xm[mask] = np.nan
                                matrices[ds][cond] = model_matrix(xm, med, channel)
                    expected_tags = {m[0] for m in models}
                    assert {p.stem for p in (dest / 'models').glob('*.joblib')} == expected_tags
                    assert {p.stem for p in (dest / 'training_masks').glob('*.npz')} == expected_tags
                    fits = {r['tag']: r for r in read_json(dest / 'fit_records.json')}
                    assert set(fits) == expected_tags
                    components = {}
                    for tag, arm, alpha, seed in models:
                        mask, expected_budget = training_mask(x, yy, arm, alpha, seed)
                        assert not (mask & missing).any()
                        with np.load(dest / 'training_masks' / f'{tag}.npz', allow_pickle=False) as z:
                            assert np.array_equal(z['new_mask'], mask), (channel, rep['id'], split, tag, 'mask')
                        cell_fingerprint.update(mask.tobytes())
                        report['checked']['training_masks'] += 1
                        class_deleted = np.stack([mask[yy == c].sum(0) for c in [0, 1]])
                        with np.load(dest / 'diagnostics' / f'{tag}.npz', allow_pickle=False) as diag:
                            assert np.array_equal(diag['row_deleted'], mask.sum(1))
                            assert np.array_equal(diag['col_deleted'], mask.sum(0))
                            assert np.array_equal(diag['class_col_deleted'], class_deleted)
                            allclose(diag['q_after'], (class_missing + class_deleted) / counts[:, None], 1e-15, 'q after')
                            err = allclose(diag['expected_col'], expected_budget, tol['budget_atol'], 'expected budget')
                            report['max_errors']['expected_budget'] = max(report['max_errors']['expected_budget'], err)
                        report['checked']['diagnostic_files'] += 1
                        modelpath = dest / 'models' / f'{tag}.joblib'
                        payload = joblib.load(modelpath)
                        for key, value in [('arm', arm), ('alpha', alpha), ('mask_seed', seed), ('channel', channel),
                                           ('repeat', rep['id']), ('model_seed', rep['model_seed']),
                                           ('config_sha256', report['config_sha256'])]:
                            assert payload[key] == value, (channel, rep['id'], split, tag, key)
                        assert np.array_equal(payload['train_rows'], tr)
                        allclose(payload['medians'], med, 0., 'model median')
                        model = payload['model']
                        assert np.array_equal(model.classes_, [0, 1])
                        assert model.n_features_in_ == (340 if channel == 'M3' else 170)
                        assert model.get_params() == params, (channel, tag, 'model params')
                        assert fits[tag]['effective_params'] == params
                        assert fits[tag]['n_iter'] == int(model.n_iter_)
                        assert fits[tag]['do_early_stopping'] == bool(model.do_early_stopping_)
                        assert fits[tag]['model_sha256'] == sha256(modelpath)
                        config_name = tag.split('_seed')[0]
                        for ds in ['val', 'test']:
                            with np.load(dest / 'predictions' / f'{tag}__{ds}.npz', allow_pickle=False) as pred:
                                assert set(pred.files) == {'row_indices', 'y', *conditions}
                                assert np.array_equal(pred['row_indices'], rowsets[ds]) and np.array_equal(pred['y'], ys[ds])
                                for cond in conditions:
                                    p = pred[cond]
                                    assert p.shape == ys[ds].shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                                    fresh = model.predict_proba(matrices[ds][cond])[:, 1]
                                    err = allclose(fresh, p, pred_atol, f'{channel}/{rep["id"]}/{split}/{tag}/{ds}/{cond}')
                                    report['max_errors']['prediction'] = max(report['max_errors']['prediction'], err)
                                    report['checked']['prediction_arrays'] += 1
                                    components.setdefault(f'{config_name}__{ds}__{cond}', []).append(p.copy())
                        report['checked']['models'] += 1
                        del model, payload, mask
                    key = f'rep{rep["id"]}/split{split}'
                    digest = cell_fingerprint.hexdigest()
                    report['pairing_fingerprints'][f'{channel}/{key}'] = digest
                    if key in pairing:
                        assert pairing[key] == digest, (key, channel, 'cross-channel inputs mismatch')
                    else:
                        pairing[key] = digest
                    metrics = read_json(dest / 'metrics.json'); calculated = {}
                    with np.load(dest / 'ensemble_predictions.npz', allow_pickle=False) as z:
                        assert set(z.files) == set(components) | {'train_rows', 'val_rows', 'test_rows', 'y_val', 'y_test'}
                        for name, arr in [('train_rows', tr), ('val_rows', va), ('test_rows', rowsets['test']),
                                          ('y_val', ys['val']), ('y_test', ys['test'])]:
                            assert np.array_equal(z[name], arr)
                        for full_key, parts in components.items():
                            name, ds, cond = full_key.split('__')
                            assert len(parts) == (1 if name == 'BASE' else 3)
                            err = allclose(z[full_key], np.mean(parts, axis=0), pred_atol, 'ensemble')
                            report['max_errors']['ensemble'] = max(report['max_errors']['ensemble'], err)
                            report['checked']['ensemble_arrays'] += 1
                            ap = independent_ap(ys[ds], z[full_key]); calculated.setdefault(name, {})[f'{ds}__{cond}'] = ap
                            err = allclose(metrics[name][f'{ds}__{cond}'], ap, ap_atol, 'AP')
                            report['max_errors']['ap'] = max(report['max_errors']['ap'], err)
                            report['checked']['ap_cells'] += 1
                    assert set(metrics) == set(calculated)
                    assert all(set(metrics[n]) == set(calculated[n]) for n in metrics)
                    selection = read_json(dest / 'selection.json')
                    assert selection['selected_before_test_prediction'] is True
                    val_metrics = {n: {c: d['val__' + c] for c in conditions} for n, d in calculated.items()}
                    choices = {}
                    for arm in ['C', 'B']:
                        chosen, candidates = independent_selection(val_metrics, arm, cfg['alphas'], cfg['s0_tolerance'], tol['selection_tie_atol'])
                        assert selection['chosen'][arm]['alpha'] == chosen['alpha']
                        assert selection['chosen'][arm]['config'] == chosen['config']
                        assert len(selection['candidates'][arm]) == len(candidates) == 4
                        for saved, fresh in zip(selection['candidates'][arm], candidates):
                            for k in ['alpha', 'config', 'eligible']:
                                assert saved[k] == fresh[k]
                            for k in ['s0', 'worst4']:
                                err = allclose(saved[k], fresh[k], ap_atol, 'selection score')
                                report['max_errors']['selection_score'] = max(report['max_errors']['selection_score'], err)
                        choices[arm] = chosen['alpha']; report['checked']['selections'] += 1
                    grouped = {}
                    for name, met in calculated.items():
                        grouped[name] = {ds: {g: met[f'{ds}__S0'] if g == 'S0' else
                                             float(np.mean([met[f'{ds}__{c}'] for c in conditions if c.startswith(g + '_')]))
                                             for g in groups} for ds in ['val', 'test']}
                    for arm in ['C', 'B']:
                        grouped[arm + '_selected'] = grouped[selection['chosen'][arm]['config']]
                    saved_summary = summary_map[(rep['id'], split)]
                    assert saved_summary['model_seed'] == rep['model_seed']
                    assert saved_summary['chosen_alphas'] == choices
                    assert set(saved_summary['ap']) == set(grouped)
                    for name in grouped:
                        for ds in ['val', 'test']:
                            for g in groups:
                                err = allclose(saved_summary['ap'][name][ds][g], grouped[name][ds][g], ap_atol, 'summary')
                                report['max_errors']['summary'] = max(report['max_errors']['summary'], err)
                                report['checked']['summary_cells'] += 1
                    report['cells'].append({'channel': channel, 'repeat': rep['id'], 'split': split, 'status': 'passed', 'selected_alphas': choices})
                    print(json.dumps({'event': 'verification_cell_complete', 'channel': channel, 'repeat': rep['id'], 'split': split,
                                      'models_checked': report['checked']['models'], 'max_errors': report['max_errors']}), flush=True)
                    del matrices, components, raw, x
        n = len(channels)
        for k, expected in [('models', 285*n), ('training_masks', 285*n), ('evaluation_masks', 208*n),
                            ('prediction_arrays', 7410*n), ('ensemble_arrays', 2730*n), ('ap_cells', 2730*n),
                            ('selections', 30*n), ('diagnostic_files', 285*n), ('summary_cells', 1350*n)]:
            assert report['checked'][k] == expected, (k, report['checked'][k], expected)
        report['paired_cross_channel_cells'] = len(pairing) if len(channels) > 1 else 0
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'; report['error_type'] = type(exc).__name__; report['error'] = str(exc)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['wall_seconds'] = time.perf_counter() - started
        write_json(output, report)
    return report


def combine_reports(output):
    """Combine completed channel reports and require identical paired input fingerprints."""
    cfg = read_json(HERE / 'config.json')
    reports = {ch: read_json(HERE / f'verification_{ch}.json') for ch in cfg['channels']}
    for ch, report in reports.items():
        assert report['status'] == 'passed' and report['channels'] == [ch]
        for key, path in [('config_sha256', 'config.json'), ('freeze_sha256', 'freeze.json'),
                          ('runner_sha256', 'run_channels.py'), ('verifier_sha256', 'verify_results.py')]:
            assert report[key] == sha256(HERE / path), (ch, key, 'report stale')
    first = reports[cfg['channels'][0]]
    checked = {key: sum(r['checked'][key] for r in reports.values()) for key in first['checked']}
    # Freeze inputs were checked separately for every channel, but count unique frozen files once.
    counts = [r['checked']['freeze_files'] for r in reports.values()]
    assert len(set(counts)) == 1
    checked['freeze_files'] = counts[0]
    maximum = {key: max(r['max_errors'][key] for r in reports.values()) for key in first['max_errors']}
    pairing = {}
    for rep in cfg['repeats']:
        for split in cfg['split_seeds']:
            key = f'rep{rep["id"]}/split{split}'
            values = [reports[ch]['pairing_fingerprints'][f'{ch}/{key}'] for ch in cfg['channels']]
            assert len(set(values)) == 1, (key, 'cross-channel mismatch')
            pairing[key] = values[0]
    for key, expected in [('models', 855), ('training_masks', 855), ('prediction_arrays', 22230),
                          ('ap_cells', 8190), ('selections', 90), ('ensemble_arrays', 8190),
                          ('summary_cells', 4050), ('diagnostic_files', 855), ('evaluation_masks', 624)]:
        assert checked[key] == expected, (key, checked[key], expected)
    report = {'status': 'passed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Composition of three full independent channel artifact audits plus exact cross-channel paired-input fingerprint equality.',
              'checked': checked, 'max_errors': maximum, 'paired_cross_channel_cells': len(pairing),
              'paired_training_mask_specs': 285, 'pairing_fingerprints': pairing,
              'channels': cfg['channels'], 'limitations': first['limitations'],
              'channel_reports': {ch: {'file': f'verification_{ch}.json',
                                       'sha256': sha256(HERE / f'verification_{ch}.json'),
                                       'wall_seconds': r['wall_seconds']} for ch, r in reports.items()},
              'verifier_sha256': sha256(__file__), 'config_sha256': first['config_sha256'],
              'freeze_sha256': first['freeze_sha256'], 'runner_sha256': first['runner_sha256'],
              'cells': [cell for r in reports.values() for cell in r['cells']]}
    assert len(pairing) == 15 and len(report['cells']) == 45
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'verification.json')
    parser.add_argument('--channel', choices=['M2', 'M3', 'MO'])
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--combine', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({'status': 'passed', 'manual_AP_examples': check_ap_examples(), 'channel_examples': check_channel_examples()}))
    else:
        if args.output.exists():
            raise SystemExit(f'Refusing to overwrite {args.output}; choose a new --output')
        if args.combine:
            assert args.channel is None, '--combine is global'
            result = combine_reports(args.output)
        else:
            with threadpool_limits(limits=2):
                result = verify(args.output, args.channel)
        print(json.dumps({'status': result['status'], 'checked': result['checked'], 'max_errors': result['max_errors']}, indent=2))
