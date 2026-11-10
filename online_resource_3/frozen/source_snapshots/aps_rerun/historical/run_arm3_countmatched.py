#!/usr/bin/env python
"""R9: full-scale strictly count-matched, class-blind control arm (Arm3).

Question (reviewer R9).  The paper's main contrast C vs B ("dose matched" class-conditional
versus class-blind masking) is matched only in *training-distribution expectation*, not
cell-by-cell / column-by-column.  Two explanations for the C-B loss therefore cannot be
separated:

    (i)  within-class allocation  (deleting the informative class inside every column), and
    (ii) column-dose profile      (deleting a different number of cells per column).

This script trains the missing arm at full scale:

    Arm3  = class-blind arm whose per-column deletion counts are EXACTLY C's realized
            per-column counts k_j^C (same mask seed), but inside column j the k_j^C cells are
            drawn uniformly at random from the currently visible cells, ignoring the label.

Then, pointwise in every metric,

    C - B   =   (C - Arm3)   +   (Arm3 - B)
                within-class      column-dose
                allocation        profile

(C - Arm3) keeps the per-column dose vector fixed at C's and changes ONLY the within-column
class allocation, so it is a pure within-class allocation effect.  (Arm3 - B) keeps the
within-column rule class-blind (uniform) for both arms and changes ONLY the per-column dose
vector, so it is a pure column-profile effect.  The decomposition is an algebraic identity;
the JSON reports its numerical residual.

Reference pipeline is reused verbatim (no re-implementation):
  * 03_.../src/aps_io.py       load_aps_csv / split_features / dev_split
  * 03_.../src/ccms.py         mask_class_conditional (to obtain C's realized counts)
  * 03_.../src/pipeline.py     MedianImputer / _make_model
  * 03_.../scripts/run_cb_firstlook.py  build_eval_inputs (13 stress inputs)
  * 02_.../scripts/run_supplement.py    aps(): matrix/fit_predict/ens3 (copied below)
  * 02_.../scripts/summarize.py::aps_boot  prep/wap/interval + bootstrap stream (copied below)

Fiducial gate (must pass before anything is published): split seed 11, mask seeds
9107/9108/9109, recompute M3(S0) and C025(ens3 S0) validation probabilities and compare with
results/aps_seed11.npz; tolerance max|diff| <= 1e-6.

Outputs (only under revision/analysis/):
  arm3_seed{11,22,33,44,55}.npz   keys: y_val,y_test,val_rows,train_rows,
                                        A3__{val,test}__<cond>   (required)
                                        A4__{val,test}__<cond>   (secondary placebo, see below)
  r9_arm3_countmatched.json
  notes_c.md (written separately)

Arm4 (secondary, clearly labelled "placebo"): the SAME class-blind, exact-count masking
mechanism as Arm3 but with the per-column counts taken from B's realized mask instead of C's.
It isolates "Bernoulli masking -> exact-count masking" as a mechanism change, i.e. it is the
placebo for Arm3's construction.  The originally sketched Arm4 ("C's class-conditional
formula rescaled to B's per-column budget") is mathematically infeasible: B's per-column
budget exceeds the number of rows in the class that C actually deletes from, because C
concentrates 100% of a column's deletions in the lower-missing class (a_cj = 0 for the
qmax class).  The infeasibility bound is recorded in the JSON instead.

Usage:
    python run_arm3_countmatched.py            # fidelity -> train -> summarize
    python run_arm3_countmatched.py fidelity   # fiducial gate only
    python run_arm3_countmatched.py train      # Arm3/Arm4 fits only
    python run_arm3_countmatched.py summarize  # metrics + bootstrap only
Env: CCMS_NBOOT (default 1000).
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score as AP
from threadpoolctl import threadpool_limits

threadpool_limits(2)

ROOT = Path(__file__).resolve().parents[2]  # bundle root
PKG = ROOT / 'CCMS论文修订完整资料包_2026-11-01_192149'
ARCHIVE = PKG / '03_原始项目归档' / 'CCMS核查_2026-10-31'
REF = PKG / '02_本轮代码与结果' / 'results'
DATA = PKG / '04_实际使用的数据' / 'aps'
OUT = ROOT / 'revision' / 'analysis'
sys.path[:0] = [str(ARCHIVE / 'src'), str(ARCHIVE / 'scripts')]

from aps_io import dev_split, load_aps_csv, split_features  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
)
from pipeline import MedianImputer, _make_model  # noqa: E402
from run_cb_firstlook import build_eval_inputs  # noqa: E402

SEEDS = [11, 22, 33, 44, 55]
MASK_SEEDS = [9107, 9108, 9109]
ALPHA = 0.25
MODEL_SEED = 2026
N_BOOT = int(os.environ.get('CCMS_NBOOT', '1000'))
BOOT_SEED = 20261101
FID_TOL = 1e-6
DRAW_TOL = 1e-12
DRAWS_PATH = OUT / 'r9_arm3_bootstrap_draws.npz'
REUSE_BOOT = os.environ.get('CCMS_REUSE_BOOT') == '1'

# families exactly as in 02_.../scripts/summarize.py (S1_10 added as extra, no reference)
FAM = {
    'S0': ['S0'],
    'S1_10': ['S1_10_101', 'S1_10_202', 'S1_10_303'],
    'S1_30': ['S1_30_101', 'S1_30_202', 'S1_30_303'],
    'S2': ['S2_1101', 'S2_1202', 'S2_1303'],
    'S3': ['S3_2101', 'S3_2202', 'S3_2303'],
}
FAM_REF = {k: v for k, v in FAM.items() if k != 'S1_10'}  # families in aps_matched_bootstrap.json
DATASETS = ['val', 'test']
PAIRS = ['C-B', 'C-A3', 'A3-B', 'A4-B', 'A3-A4']


# ---------------------------------------------------------------------------
# helpers copied verbatim from 02_.../scripts/run_supplement.py::aps
# ---------------------------------------------------------------------------
def matrix(x, imp, bb='M3'):
    return x if bb == 'M4' else np.hstack([imp.transform(x), np.isnan(x).astype(float)])


def fit_predict(x, y, imp, evals, bb='M3'):
    m = _make_model(MODEL_SEED)
    m.fit(matrix(x, imp, bb), y)
    return {k: m.predict_proba(matrix(v, imp, bb))[:, 1] for k, v in evals.items()}


# ---------------------------------------------------------------------------
# helpers copied verbatim from 02_.../scripts/summarize.py::aps_boot
# ---------------------------------------------------------------------------
def prep(y, p):
    o = np.argsort(-p, kind='stable')
    last = np.r_[np.flatnonzero(np.diff(p[o]) != 0), len(y) - 1]
    return o, y[o], last


def wap(pr, w):
    o, y, last = pr
    ww = w[o]
    tp = np.cumsum(ww * y)[last]
    den = np.cumsum(ww)[last]
    total = tp[-1]
    if total <= 0:
        raise ValueError('zero bootstrap positives')
    return float(np.sum(np.diff(np.r_[0., tp]) * np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)) / total)


def interval(v):
    return [float(z) for z in np.quantile(v, [.025, .975])]


def metrics_of(y, preds_by_cond):
    return float(np.mean([AP(y, preds_by_cond[k]) for k in preds_by_cond]))


# ---------------------------------------------------------------------------
# Arm3 / Arm4 mask builders
# ---------------------------------------------------------------------------
def mask_count_matched(x, kj, seed):
    """Class-blind mask deleting exactly kj[j] visible cells in each column j.

    Within column j the kj[j] cells are drawn uniformly at random from the currently
    visible cells, independently of the label.  RNG = np.random.default_rng(seed),
    drawn column by column in ascending column order (deterministic, unrelated to the
    uniform stream that produced C's mask).
    """
    n, p = x.shape
    visible = ~np.isnan(x)
    mask = np.zeros((n, p), dtype=bool)
    rng = np.random.default_rng(seed)
    truncated = []
    for j in range(p):
        k = int(kj[j])
        if k <= 0:
            continue
        idx = np.flatnonzero(visible[:, j])
        if k > len(idx):
            truncated.append({'column': int(j), 'requested': k, 'available': int(len(idx))})
            k = len(idx)
        mask[rng.choice(idx, size=k, replace=False), j] = True
    return mask, {'truncated': truncated}


def apply_mask(x, mask):
    out = x.copy()
    out[mask] = np.nan
    return out


def fidelity(env):
    """Fiducial gate: reproduce results/aps_seed11.npz M3 / C025 val+test S0, seed 11."""
    print('[fid] loading data', flush=True)
    df, meta = load_aps_csv(DATA / 'aps_failure_training_set.csv')
    X, y = split_features(df)
    dt, mt = load_aps_csv(DATA / 'aps_failure_test_set.csv')
    Xt, yt = split_features(dt)
    sp = dev_split(y, seed=11, stratify=True)
    tr, va = sp['train'], sp['val']
    x, yy = X[tr], y[tr]
    imp = MedianImputer().fit(x)
    ev = {f'val__{k}': v[0] for k, v in build_eval_inputs(X[va]).items()}
    ev.update({f'test__{k}': v[0] for k, v in build_eval_inputs(Xt).items()})
    ref = np.load(REF / 'aps_seed11.npz')

    t = time.perf_counter()
    p_m3 = fit_predict(x, yy, imp, ev)
    sec_m3 = time.perf_counter() - t
    out = {
        'split_seed': 11,
        'runs': [{'config': 'M3', 'mask_seed': None, 'seconds': round(sec_m3, 3),
                  'max_abs_diff': {
                      'val__S0': float(np.abs(p_m3['val__S0'] - ref['M3__val__S0']).max()),
                      'test__S0': float(np.abs(p_m3['test__S0'] - ref['M3__test__S0']).max())}}],
    }
    runs, secs = [], []
    for ms in MASK_SEEDS:
        xm, info = mask_class_conditional(x, yy, ALPHA, seed=ms)
        t = time.perf_counter()
        runs.append(fit_predict(xm, yy, imp, ev))
        secs.append(time.perf_counter() - t)
        out['runs'].append({'config': 'C025', 'mask_seed': ms, 'seconds': round(secs[-1], 3),
                            'deleted_total': int(info['new_mask'].sum())})
        print(f'[fid] C mask_seed {ms} fit {secs[-1]:.2f}s', flush=True)
    ens = {k: np.mean([r[k] for r in runs], 0) for k in ev}
    d_val = float(np.abs(ens['val__S0'] - ref['C025__val__S0']).max())
    d_test = float(np.abs(ens['test__S0'] - ref['C025__test__S0']).max())
    d_val_s130 = float(np.abs(ens['val__S1_30_101'] - ref['C025__val__S1_30_101']).max())
    out['runs'].append({'config': 'C025(ens3)', 'mask_seed': MASK_SEEDS,
                        'max_abs_diff': {'val__S0': d_val, 'test__S0': d_test,
                                         'val__S1_30_101': d_val_s130}})
    out['C025_ens3_max_abs_diff'] = {'val__S0': d_val, 'test__S0': d_test, 'val__S1_30_101': d_val_s130}
    all_times = [r['seconds'] for r in out['runs'] if 'seconds' in r]
    out['fit_seconds'] = {'m3': round(sec_m3, 3), 'C025_single_seed': [round(s, 3) for s in secs],
                          'n_fits': len(all_times), 'mean': round(float(np.mean(all_times)), 3),
                          'min': round(float(np.min(all_times)), 3),
                          'max': round(float(np.max(all_times)), 3)}
    out['tolerance'] = FID_TOL
    worst = max([v for r in out['runs'] for v in r.get('max_abs_diff', {}).values()])
    out['max_abs_diff_overall'] = float(worst)
    out['pass'] = bool(worst <= FID_TOL)
    out['environment'] = env
    out['n_train'] = int(len(tr))
    out['n_val'] = int(len(va))
    out['n_test'] = int(len(yt))
    return out


def manipulation_check(env):
    """Mask-only diagnostics (no fitting): did Arm3 reproduce B's within-column class
    composition, and is C's within-column allocation really class-conditional?

    For each split / mask seed we build C, B and Arm3 masks exactly as in training and count
    deletions by class.  This is cheap (no model fits) and reports the mechanism directly.
    """
    print('[manip] loading data', flush=True)
    df, meta = load_aps_csv(DATA / 'aps_failure_training_set.csv')
    X, y = split_features(df)
    out = {'environment': env, 'alpha': ALPHA, 'mask_seeds': MASK_SEEDS,
           'note': 'mask-level diagnostics only; no model refitting. Rates are deleted visible '
                   'cells / visible cells of that class over the whole training matrix.',
           'splits': []}
    for seed in SEEDS:
        sp = dev_split(y, seed=seed, stratify=True)
        tr = sp['train']
        x, yy = X[tr], y[tr]
        q, counts, _ = estimate_class_rates(x, yy)
        bvec = class_blind_rates(q, counts, ALPHA)
        vis = np.stack([(~np.isnan(x[yy == c])).sum(0) for c in [0, 1]])
        rows = []
        for ms in MASK_SEEDS:
            _, info_c = mask_class_conditional(x, yy, ALPHA, seed=ms)
            _, info_b = mask_class_blind(x, bvec, seed=ms)
            m3, _ = mask_count_matched(x, info_c['deleted_per_col'].astype(int), seed=ms * 1000 + 7)
            m4, _ = mask_count_matched(x, info_b['deleted_per_col'].astype(int), seed=ms * 1000 + 11)
            assert np.array_equal(m4.sum(0), info_b['deleted_per_col'].astype(int))
            rec = {'mask_seed': ms,
                   'A4_per_column_counts_equal_B': bool(
                       np.array_equal(m4.sum(0), info_b['new_mask'].sum(0))),
                   'A4_vs_B_max_abs_per_column_diff': int(
                       np.abs(m4.sum(0) - info_b['new_mask'].sum(0)).max()),
                   'A3_vs_A4_max_abs_per_column_diff': int(np.abs(m3.sum(0) - m4.sum(0)).max())}
            for tag, mask in [('C', info_c['new_mask']), ('B', info_b['new_mask']),
                              ('A3', m3), ('A4', m4)]:
                by_class = [int(mask[yy == c].sum()) for c in [0, 1]]
                rec[tag] = {
                    'deleted_by_class': by_class,
                    'deleted_total': int(mask.sum()),
                    'deletion_rate_within_class': [float(by_class[c] / vis[c].sum()) for c in [0, 1]],
                    'positive_to_negative_rate_ratio': float((by_class[1] / vis[1].sum()) /
                                                             (by_class[0] / vis[0].sum())),
                }
            rec['A3_minus_B_positive_class_rate'] = (
                rec['A3']['deletion_rate_within_class'][1] - rec['B']['deletion_rate_within_class'][1])
            rec['C_minus_B_positive_class_rate'] = (
                rec['C']['deletion_rate_within_class'][1] - rec['B']['deletion_rate_within_class'][1])
            rows.append(rec)
        out['splits'].append({
            'seed': seed, 'counts': [int(c) for c in counts],
            'visible_by_class': [int(v) for v in vis.sum(1)],
            'n_columns_where_class1_is_qmax': int((np.argmax(q, axis=0) == 1).sum()),
            'mask_seeds': rows,
        })
        print(f'[manip] seed {seed} done', flush=True)
    return out


def train_arms(env):
    """Fit Arm3 (required) and Arm4 (secondary placebo) for all 5 splits x 3 mask seeds."""
    print('[train] loading data', flush=True)
    df, meta = load_aps_csv(DATA / 'aps_failure_training_set.csv')
    X, y = split_features(df)
    dt, mt = load_aps_csv(DATA / 'aps_failure_test_set.csv')
    Xt, yt = split_features(dt)
    report = {'environment': env, 'alpha': ALPHA, 'mask_seeds': MASK_SEEDS,
              'note': 'A3 = per-column counts exactly equal to C, uniform within column (class-blind). '
                      'A4 = same exact-count class-blind mechanism, per-column counts taken from B.',
              'splits': []}
    for seed in SEEDS:
        t_split = time.perf_counter()
        sp = dev_split(y, seed=seed, stratify=True)
        tr, va = sp['train'], sp['val']
        x, yy = X[tr], y[tr]
        imp = MedianImputer().fit(x)
        ev = {f'val__{k}': v[0] for k, v in build_eval_inputs(X[va]).items()}
        ev.update({f'test__{k}': v[0] for k, v in build_eval_inputs(Xt).items()})
        q, counts, _ = estimate_class_rates(x, yy)
        bvec = class_blind_rates(q, counts, ALPHA)
        vis = (~np.isnan(x)).sum(0)
        exp_c = ALPHA * sum(counts[c] * (q.max(0) - q[c]) for c in [0, 1])
        exp_b = bvec * vis
        diag = {'expected_per_column_max_abs_diff_C_vs_B': float(np.abs(exp_c - exp_b).max()),
                'mask_seeds': []}
        runs = {'A3': [], 'A4': []}
        costs = []
        for ms in MASK_SEEDS:
            # --- C's realized per-column deletion counts (same mask seed) ---
            _, info_c = mask_class_conditional(x, yy, ALPHA, seed=ms)
            kj_c = info_c['deleted_per_col'].astype(int)
            # --- B's realized per-column deletion counts (same mask seed) ---
            _, info_b = mask_class_blind(x, bvec, seed=ms)
            kj_b = info_b['deleted_per_col'].astype(int)

            # --- Arm3: exact C counts, uniform within column ---
            m3, tr3 = mask_count_matched(x, kj_c, seed=ms * 1000 + 7)
            k3 = m3.sum(0)
            assert np.array_equal(k3, kj_c), (seed, ms, 'Arm3 count mismatch')
            # feasibility bound for the infeasible "class-conditional at B's budget" variant
            vis_pos = int((~np.isnan(x[yy == 1])).sum(0).max())
            vis_neg = int((~np.isnan(x[yy == 0])).sum(0).max())
            t0 = time.perf_counter()
            runs['A3'].append(fit_predict(apply_mask(x, m3), yy, imp, ev))
            costs.append(time.perf_counter() - t0)

            # --- Arm4 (placebo): exact B counts, uniform within column ---
            m4, tr4 = mask_count_matched(x, kj_b, seed=ms * 1000 + 11)
            k4 = m4.sum(0)
            assert np.array_equal(k4, kj_b), (seed, ms, 'Arm4 count mismatch')
            t0 = time.perf_counter()
            runs['A4'].append(fit_predict(apply_mask(x, m4), yy, imp, ev))
            costs.append(time.perf_counter() - t0)

            # infeasibility of class-conditional allocation at B's per-column budget:
            # C deletes only from the lower-missing class, whose visible row count per column
            # bounds the achievable class-conditional budget.
            vis_by_class = np.stack([(~np.isnan(x[yy == c])).sum(0) for c in [0, 1]])
            qmax_cls = np.argmax(q, axis=0)
            low_cls = 1 - qmax_cls
            cap = vis_by_class[low_cls, np.arange(x.shape[1])]
            n_over = int((kj_b > cap).sum())
            diag['mask_seeds'].append({
                'mask_seed': ms,
                'C_realized_deleted_total': int(kj_c.sum()),
                'B_realized_deleted_total': int(kj_b.sum()),
                'A3_deleted_total': int(k3.sum()),
                'A4_deleted_total': int(k4.sum()),
                'max_abs_per_column_A3_minus_C': int(np.abs(k3 - kj_c).max()),
                'per_column_counts_equal_A3_C': bool(np.array_equal(k3, kj_c)),
                'n_columns_with_any_deletion_C': int((kj_c > 0).sum()),
                'n_columns_with_any_deletion_B': int((kj_b > 0).sum()),
                'max_abs_per_column_C_minus_B': int(np.abs(kj_c - kj_b).max()),
                'mean_abs_per_column_C_minus_B': float(np.abs(kj_c - kj_b).mean()),
                'A3_truncation': tr3['truncated'],
                'A4_truncation': tr4['truncated'],
                'class_conditional_at_B_budget_infeasible_columns': n_over,
                'class_conditional_at_B_budget_max_budget_over_capacity': float(
                    np.max(kj_b - cap)) if n_over else 0.0,
                'deleted_per_col_C': kj_c.tolist(),
                'deleted_per_col_B': kj_b.tolist(),
            })
            print(f'[train] seed {seed} mask_seed {ms} A3/A4 fits '
                  f'{costs[-2]:.1f}s/{costs[-1]:.1f}s '
                  f'deleted C={kj_c.sum()} A3={k3.sum()} B={kj_b.sum()} A4={k4.sum()}', flush=True)

        payload = {'y_val': y[va], 'y_test': yt, 'val_rows': va, 'train_rows': tr}
        ens = {}
        for tag, arr in runs.items():
            ens[tag] = {k: np.mean([r[k] for r in arr], 0) for k in ev}
            for k, p in ens[tag].items():
                payload[f'{tag}__{k}'] = p
        np.savez_compressed(OUT / f'arm3_seed{seed}.npz', **payload)
        report['splits'].append({
            'seed': seed, 'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(yt)),
            'pos_train': int(yy.sum()), 'pos_val': int(y[va].sum()), 'pos_test': int(yt.sum()),
            'fit_seconds': {'n_fits': len(costs), 'mean': round(float(np.mean(costs)), 3),
                            'list': [round(c, 3) for c in costs]},
            'count_matching': diag,
            'seconds_total': round(time.perf_counter() - t_split, 1),
        })
        print(f'[train] seed {seed} saved, total {time.perf_counter() - t_split:.1f}s', flush=True)
    return report


def summarize(env, fid, train_report, manip):
    """Point estimates + paired bootstrap decomposition with the reference RNG stream."""
    packs = [np.load(REF / f'aps_seed{s}.npz') for s in SEEDS]
    arms = [np.load(OUT / f'arm3_seed{s}.npz') for s in SEEDS]
    for s, a in zip(SEEDS, arms):
        assert np.array_equal(a['val_rows'], packs[SEEDS.index(s)]['val_rows'])
    ref_json = json.loads((REF / 'aps_matched_bootstrap.json').read_text())
    ref_draws = np.load(REF / 'aps_bootstrap_draws.npz')

    # sanity: identical row order / labels between reference and this run
    label_check = max(float(np.abs(p['y_val'] - a['y_val']).max()) for p, a in zip(packs, arms))
    label_check = max(label_check, max(float(np.abs(p['y_test'] - a['y_test']).max()) for p, a in zip(packs, arms)))

    results = {}
    draws_out = {}
    ref_check = {'pass': True, 'checks': []}
    for dataset in DATASETS:
        ys = [z['y_' + dataset] for z in packs]
        ids = [z['val_rows'] for z in packs] if dataset == 'val' else [np.arange(len(ys[0])) for _ in packs]
        union = np.unique(np.concatenate(ids))
        uy = np.empty(len(union), int)
        mappings = [np.searchsorted(union, v) for v in ids]
        for y, m in zip(ys, mappings):
            uy[m] = y

        # ---------------- point estimates (unweighted sklearn AP) ----------------
        aps = {}
        pr = {}
        points = {f: {p: np.zeros(5) for p in PAIRS} for f in FAM}
        for f, conds in FAM.items():
            pr[f] = []
            for si, (y, z, a) in enumerate(zip(ys, packs, arms)):
                row = []
                for k in conds:
                    vals = {
                        'C': float(AP(y, z['C025__' + dataset + '__' + k])),
                        'B': float(AP(y, z['B025__' + dataset + '__' + k])),
                        'M3': float(AP(y, z['M3__' + dataset + '__' + k])),
                        'A3': float(AP(y, a['A3__' + dataset + '__' + k])),
                        'A4': float(AP(y, a['A4__' + dataset + '__' + k])),
                    }
                    aps[(f, si, k)] = vals
                    row.append({'C': prep(y, z['C025__' + dataset + '__' + k]),
                                'B': prep(y, z['B025__' + dataset + '__' + k]),
                                'M3': prep(y, z['M3__' + dataset + '__' + k]),
                                'A3': prep(y, a['A3__' + dataset + '__' + k]),
                                'A4': prep(y, a['A4__' + dataset + '__' + k])})
                pr[f].append(row)
            for pair in PAIRS:
                ca, cb = pair.split('-')
                points[f][pair] = np.array([
                    float(np.mean([aps[(f, si, k)][ca] - aps[(f, si, k)][cb] for k in conds]))
                    for si in range(5)])

        # ---------------- paired bootstrap, RNG stream identical to aps_boot ----------------
        boots = {f: {p: np.zeros((N_BOOT, 5)) for p in PAIRS} for f in FAM}
        if REUSE_BOOT and DRAWS_PATH.exists():
            saved = np.load(DRAWS_PATH)
            for f in FAM:
                for pair in PAIRS:
                    boots[f][pair] = saved[f'{dataset}__{f}__{pair}']
            print(f'[sum] dataset={dataset} bootstrap draws reused from {DRAWS_PATH.name}', flush=True)
            rng = None
        else:
            rng = np.random.default_rng(BOOT_SEED)  # reset once per dataset (as in aps_boot)
        pidx = np.flatnonzero(uy == 1)
        nidx = np.flatnonzero(uy == 0)
        t0 = time.perf_counter()
        for b in (range(N_BOOT) if rng is not None else []):
            ix = np.r_[rng.choice(pidx, len(pidx), replace=True),
                       rng.choice(nidx, len(nidx), replace=True)]
            weights = np.bincount(ix, minlength=len(union)).astype(float)
            for f in FAM:
                for si, m in enumerate(mappings):
                    w = weights[m]
                    acc = {p: [] for p in PAIRS}
                    for row in pr[f][si]:
                        vals = {cfg: wap(row[cfg], w) for cfg in row}
                        for pair in PAIRS:
                            ca, cb = pair.split('-')
                            acc[pair].append(vals[ca] - vals[cb])
                    for pair in PAIRS:
                        boots[f][pair][b, si] = np.mean(acc[pair])
            if b % 250 == 0:
                print(f'[sum] dataset={dataset} boot={b} {time.perf_counter() - t0:.1f}s', flush=True)

        # ---------------- aggregate ----------------
        results[dataset] = {}
        for f in FAM:
            cell = {'per_config_mean_AP': {}, 'per_config_per_split_AP': {}, 'comparisons': {}}
            for cfg in ['M3', 'C025', 'B025', 'A3', 'A4']:
                key = {'C025': 'C', 'B025': 'B'}.get(cfg, cfg)
                ps = [float(np.mean([aps[(f, si, k)][key] for k in FAM[f]])) for si in range(5)]
                cell['per_config_per_split_AP'][cfg] = ps
                cell['per_config_mean_AP'][cfg] = float(np.mean(ps))
            for pair in PAIRS:
                d = {
                    'delta_mean': float(np.mean(points[f][pair])),
                    'ci95': interval(boots[f][pair].mean(1)),
                    'per_split': [{'seed': s, 'delta': float(points[f][pair][i]),
                                   'ci95': interval(boots[f][pair][:, i])}
                                  for i, s in enumerate(SEEDS)],
                }
                cell['comparisons'][pair] = d
                draws_out[f'{dataset}__{f}__{pair}'] = boots[f][pair]
            cb = cell['comparisons']['C-B']['delta_mean']
            ca = cell['comparisons']['C-A3']['delta_mean']
            ab = cell['comparisons']['A3-B']['delta_mean']
            ra = cell['comparisons']['A4-B']['delta_mean']
            a34 = cell['comparisons']['A3-A4']['delta_mean']
            cell['decomposition'] = {
                'identity_C-B_equals_(C-A3)+(A3-B)': float(cb - (ca + ab)),
                'within_class_allocation_C_minus_A3': ca,
                'column_profile_A3_minus_B': ab,
                'within_class_share_of_C_minus_B': (ca / cb) if cb != 0 else None,
                'column_profile_share_of_C_minus_B': (ab / cb) if cb != 0 else None,
                'abs_ratio_profile_over_within_class': (abs(ab) / abs(ca)) if ca != 0 else None,
                'C_minus_A3_over_C_minus_B': (ca / cb) if cb != 0 else None,
                'A3_over_B_mean_AP_gap': cell['per_config_mean_AP']['A3'] - cell['per_config_mean_AP']['B025'],
                'C_over_A3_mean_AP_gap': cell['per_config_mean_AP']['C025'] - cell['per_config_mean_AP']['A3'],
                'placebo_A4_minus_B': ra,
                'placebo_to_effect_ratio_A4_minus_B_over_C_minus_B': (ra / cb) if cb != 0 else None,
                'pure_column_profile_A3_minus_A4': a34,
                'pure_column_profile_share_of_C_minus_B': (a34 / cb) if cb != 0 else None,
                'identity_A3-B_equals_(A3-A4)+(A4-B)': float(ab - (a34 + ra)),
                'A3_minus_B_minus_A3_minus_A4_mechanism_placebo': float(ab - a34),
            }
            results[dataset][f] = cell

        # ---------------- hard check against the saved reference bootstrap ----------------
        for f, conds in FAM_REF.items():
            r = ref_json['results'][dataset][f]
            mine = results[dataset][f]['comparisons']['C-B']
            dr = float(np.abs(np.mean(points[f]['C-B']) - r['delta_mean']))
            dci = float(np.abs(np.array(mine['ci95']) - np.array(r['ci95'])).max())
            dper = float(np.abs(np.array([z['delta'] for z in mine['per_split']]) -
                                np.array([z['delta'] for z in r['per_split']])).max())
            ddraw = float(np.abs(boots[f]['C-B'] - ref_draws[f'{dataset}__{f}']).max())
            ok = (dr <= 1e-6) and (dci <= 1e-6) and (dper <= 1e-6) and (ddraw <= DRAW_TOL)
            ref_check['pass'] = ref_check['pass'] and ok
            ref_check['checks'].append({'dataset': dataset, 'family': f,
                                        'delta_mean_abs_diff': dr,
                                        'ci95_abs_diff': dci,
                                        'per_split_delta_abs_diff': dper,
                                        'bootstrap_draws_abs_diff': ddraw,
                                        'pass': bool(ok)})

    # ---------------- headline numbers ----------------
    head = {}
    for dataset in DATASETS:
        for f in FAM:
            c = results[dataset][f]['comparisons']
            head[f'{dataset}__{f}'] = {
                'C-B': c['C-B']['delta_mean'], 'C-A3': c['C-A3']['delta_mean'],
                'A3-B': c['A3-B']['delta_mean'], 'A4-B': c['A4-B']['delta_mean'],
                'C-B_ci95': c['C-B']['ci95'], 'C-A3_ci95': c['C-A3']['ci95'],
                'A3-B_ci95': c['A3-B']['ci95'], 'A4-B_ci95': c['A4-B']['ci95'],
                'A3-A4': c['A3-A4']['delta_mean'], 'A3-A4_ci95': c['A3-A4']['ci95'],
                'A3-A4_per_split': [z['delta'] for z in c['A3-A4']['per_split']],
                'identity_residual': results[dataset][f]['decomposition'][
                    'identity_C-B_equals_(C-A3)+(A3-B)'],
                'mean_AP': results[dataset][f]['per_config_mean_AP'],
            }

    count_assert = {'all_per_column_counts_equal': True, 'max_abs_diff': 0}
    for s in train_report['splits']:
        for ms in s['count_matching']['mask_seeds']:
            count_assert['all_per_column_counts_equal'] &= ms['per_column_counts_equal_A3_C']
            count_assert['max_abs_diff'] = max(count_assert['max_abs_diff'],
                                               ms['max_abs_per_column_A3_minus_C'])

    out = {
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'python': env,
        'task': 'R9 Arm3: strictly count-matched class-blind control arm, full scale',
        'sources': sources(),
        'design': {
            'alpha': ALPHA, 'splits': SEEDS, 'mask_seeds': MASK_SEEDS,
            'model': 'HistGradientBoostingClassifier(random_state=2026), sklearn defaults',
            'M3_backbone': 'median impute (fit on unmasked train) + full missing flags',
            'ens3': 'probabilities averaged over the 3 mask seeds per arm (same as run_supplement.aps)',
            'Arm3': 'per-column deletion counts exactly equal to C\'s realized counts for the same mask '
                    'seed; cells within each column drawn uniformly among visible rows (label-blind); '
                    'RNG np.random.default_rng(mask_seed*1000+7)',
            'Arm4_placebo': 'same exact-count class-blind mechanism as Arm3, per-column counts taken '
                            'from B\'s realized mask (RNG mask_seed*1000+11); isolates the Bernoulli vs '
                            'exact-count mechanism change',
            'A3-A4_pair': 'A3 and A4 share the identical exact-count class-blind construction; only the '
                          'per-column count source differs (C vs B). A3-A4 is therefore the pure column-dose '
                          'profile effect with the Bernoulli-vs-exact-count mechanism difference removed.',
            'decomposition': 'C-B = (C-Arm3) + (Arm3-B); C-Arm3 = within-class allocation effect at fixed '
                             'column dose; Arm3-B = column-dose profile effect at fixed uniform-within-'
                             'column rule',
            'estimator': 'per split: mean over the 3 masks of (AP_a - AP_b); family: mean over 5 splits; '
                         'CI: 2.5/97.5 percentile of 1000 paired bootstrap draws, identical RNG stream as '
                         'summarize.py::aps_boot (label-stratified, shared weights across splits)',
        },
        'data': {'train_csv': str(DATA / 'aps_failure_training_set.csv'),
                 'test_csv': str(DATA / 'aps_failure_test_set.csv'),
                 'n_train_rows': 42000, 'n_val_rows': 9000, 'n_test_rows': 16000,
                 'n_features': 170, 'positives_train': 1000},
        'fidelity_check': fid,
        'count_matching_assertion': {
            'requirement': 'per-column deletion counts of Arm3 identical to C (same mask seed)',
            'all_per_column_counts_equal': bool(count_assert['all_per_column_counts_equal']),
            'max_abs_per_column_diff': count_assert['max_abs_diff'],
            'per_split': train_report['splits'] and [
                {'seed': s['seed'],
                 'C_deleted_total': [m['C_realized_deleted_total'] for m in s['count_matching']['mask_seeds']],
                 'A3_deleted_total': [m['A3_deleted_total'] for m in s['count_matching']['mask_seeds']],
                 'B_deleted_total': [m['B_realized_deleted_total'] for m in s['count_matching']['mask_seeds']],
                 'A4_deleted_total': [m['A4_deleted_total'] for m in s['count_matching']['mask_seeds']],
                 'expected_per_column_max_abs_diff_C_vs_B':
                     s['count_matching']['expected_per_column_max_abs_diff_C_vs_B'],
                 'max_abs_per_column_C_minus_B': [m['max_abs_per_column_C_minus_B'] for m in s['count_matching']['mask_seeds']],
                 'mean_abs_per_column_C_minus_B': [m['mean_abs_per_column_C_minus_B'] for m in s['count_matching']['mask_seeds']],
                 'n_columns_with_any_deletion_C': [m['n_columns_with_any_deletion_C'] for m in s['count_matching']['mask_seeds']],
                 'n_columns_with_any_deletion_B': [m['n_columns_with_any_deletion_B'] for m in s['count_matching']['mask_seeds']],
                 'class_conditional_at_B_budget_infeasible_columns':
                     [m['class_conditional_at_B_budget_infeasible_columns'] for m in s['count_matching']['mask_seeds']],
                 }
                for s in train_report['splits']],
            'note': 'B has the same per-column expected deletion count as C (identity: '
                    'n_total*(qmax-qbar) = sum_c n_c*(qmax-q_c)); C and B therefore differ in the '
                    'within-column class composition, not in the column-dose profile in expectation.',
        },
        'manipulation_check': manip,
        'arm4_count_matching_assertion': {
            'requirement': 'per-column deletion counts of Arm4 identical to B (same mask seed)',
            'all_per_column_counts_equal': all(
                m['A4_per_column_counts_equal_B']
                for sp_ in manip['splits'] for m in sp_['mask_seeds']),
            'max_abs_per_column_diff': max(
                m['A4_vs_B_max_abs_per_column_diff']
                for sp_ in manip['splits'] for m in sp_['mask_seeds']),
            'A3_vs_A4_max_abs_per_column_diff': max(
                m['A3_vs_A4_max_abs_per_column_diff']
                for sp_ in manip['splits'] for m in sp_['mask_seeds']),
        },
        'train_report': train_report,
        'results': results,
        'headline': head,
        'reference_reproduction': {
            'target': 'results/aps_matched_bootstrap.json + results/aps_bootstrap_draws.npz (C025-B025)',
            'pass': bool(ref_check['pass']),
            'checks': ref_check['checks'],
            'label_row_alignment_abs_diff': label_check,
        },
        'bootstrap_draws_file': str(DRAWS_PATH),
        'n_bootstrap': N_BOOT,
        'bootstrap_seed': BOOT_SEED,
    }
    if not REUSE_BOOT:
        np.savez_compressed(DRAWS_PATH, **draws_out)
    return out


def export_arm4():
    """Write arm4_seed{seed}.npz (structurally identical to arm3_seed*.npz, A4 keys only).

    No refitting: the A4 probabilities were already produced and saved inside
    arm3_seed{seed}.npz; this only re-packages them under the requested filename/keys.
    """
    written = []
    for seed in SEEDS:
        src = np.load(OUT / f'arm3_seed{seed}.npz')
        payload = {k: src[k] for k in ('y_val', 'y_test', 'val_rows', 'train_rows')}
        for k in src.files:
            if k.startswith('A4__'):
                payload[k] = src[k]
        assert len([k for k in payload if k.startswith('A4__')]) == 26, 'expected 13 val + 13 test A4 keys'
        path = OUT / f'arm4_seed{seed}.npz'
        np.savez_compressed(path, **payload)
        written.append(str(path))
        print(f'[export4] {path.name}: {len(payload)} arrays', flush=True)
    return written


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    keep = [
        DATA / 'aps_failure_training_set.csv',
        DATA / 'aps_failure_test_set.csv',
        ARCHIVE / 'src' / 'aps_io.py',
        ARCHIVE / 'src' / 'ccms.py',
        ARCHIVE / 'src' / 'pipeline.py',
        ARCHIVE / 'scripts' / 'run_cb_firstlook.py',
        PKG / '02_本轮代码与结果' / 'scripts' / 'run_supplement.py',
        PKG / '02_本轮代码与结果' / 'scripts' / 'summarize.py',
        REF / 'aps_matched_bootstrap.json',
        REF / 'aps_bootstrap_draws.npz',
    ]
    keep += [REF / f'aps_seed{s}.npz' for s in SEEDS]
    keep += [OUT / f'arm3_seed{s}.npz' for s in SEEDS]
    keep += [OUT / f'arm4_seed{s}.npz' for s in SEEDS]
    keep += [Path(__file__)]
    return [{'path': str(p), 'sha256': sha256(p), 'n_bytes': p.stat().st_size}
            for p in keep if p.exists()]


def env_metadata():
    return {
        'version': sys.version.split()[0],
        'version_full': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
        'threads': 2,
        'script_sha256': sha256(Path(__file__)),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    env = env_metadata()
    fid_path = OUT / 'r9_arm3_fidelity.json'
    tr_path = OUT / 'r9_arm3_train_report.json'
    manip_path = OUT / 'r9_arm3_manipulation_check.json'
    if mode in ('all', 'fidelity'):
        fid = fidelity(env)
        fid_path.write_text(json.dumps(fid, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"[fid] PASS={fid['pass']} max|diff|={fid['max_abs_diff_overall']:.3e} "
              f"fit_seconds={fid['fit_seconds']}", flush=True)
        if not fid['pass']:
            print('[fid] FIDELITY GATE FAILED - refusing to train/publish', file=sys.stderr, flush=True)
            return 2
        if mode == 'fidelity':
            return 0
    else:
        fid = json.loads(fid_path.read_text())
    if mode in ('all', 'train'):
        tr = train_arms(env)
        tr_path.write_text(json.dumps(tr, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        tr = json.loads(tr_path.read_text())
    if mode == 'export4':
        export_arm4()
        return 0
    if mode in ('all', 'manipulation'):
        manip = manipulation_check(env)
        manip_path.write_text(json.dumps(manip, ensure_ascii=False, indent=2, allow_nan=False))
        if mode == 'manipulation':
            return 0
    else:
        manip = json.loads(manip_path.read_text())
    out = summarize(env, fid, tr, manip)
    (OUT / 'r9_arm3_countmatched.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    if not out['reference_reproduction']['pass']:
        print('[sum] REFERENCE REPRODUCTION CHECK FAILED', file=sys.stderr, flush=True)
        return 2
    for key in ['val__S1_30', 'test__S1_30']:
        h = out['headline'][key]
        print(f"{key}: C-B={h['C-B']:+.4f} C-A3={h['C-A3']:+.4f} A3-B={h['A3-B']:+.4f} "
              f"resid={h['identity_residual']:.2e} | AP C={h['mean_AP']['C025']:.4f} "
              f"A3={h['mean_AP']['A3']:.4f} B={h['mean_AP']['B025']:.4f}", flush=True)
    print('[sum] wrote', OUT / 'r9_arm3_countmatched.json', flush=True)
    export_arm4()
    return 0


if __name__ == '__main__':
    sys.exit(main())
