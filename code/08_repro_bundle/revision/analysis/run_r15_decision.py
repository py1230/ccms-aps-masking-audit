#!/usr/bin/env python
"""A3 / R15: decision-level metrics for a fixed alert budget (AP -> operations bridge).

For M3 / C025 / B025, on val and the official test set, per pressure family
(S0, S1_30, S2, S3):
  * inside a family the three stress replicas are kept separate: every metric is
    computed per replica and then averaged over the three replicas (scores are
    never averaged); the family value is then averaged over the five splits;
  * recall / precision / alert counts for alert rate rho in
    {0.005, 0.01, 0.02, 0.05, 0.10} with budget = ceil(rho * N) top-scored rows;
  * recall / precision when rho is fixed at the split's own prevalence P/N
    ("budget = P alerts"), where precision equals recall by construction;
  * the alert budget (as a fraction of rows) needed to reach recall = 0.5;
  * paired bootstrap 95% CI of the recall difference C-B (primary), plus C-M3 and
    B-M3 (supplementary) at rho = 0.02 and rho = prevalence, using the EXACT A1 /
    summarize.py resampling scheme (val: unique val_rows union + searchsorted
    mapping + shared weights; test: arange IDs; rng=np.random.default_rng(20261101)
    reset once per dataset; 1000 replicates; pos then neg rng.choice).

The bootstrap recall at a fixed alert rate is the weighted analogue of the point
estimator: rows follow the fixed (weight-independent) stable score order, and the
first row whose cumulative resample weight reaches rho * total weight is taken
(row-level granularity, which reproduces the plain top-ceil(rho*N) rule exactly
under unit weights, ties included).

Inputs : results/aps_seed{11,22,33,44,55}.npz (read-only)
Output : revision/analysis/r15_decision_metrics.json
Re-run  : python run_r15_decision.py
"""
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # bundle root
PKG = ROOT / 'CCMS论文修订完整资料包_2026-11-01_192149' / '02_本轮代码与结果'
R = PKG / 'results'
OUT = ROOT / 'revision' / 'analysis'

SEEDS = [11, 22, 33, 44, 55]
FAM = {
    'S0': ['S0'],
    'S1_30': ['S1_30_101', 'S1_30_202', 'S1_30_303'],
    'S2': ['S2_1101', 'S2_1202', 'S2_1303'],
    'S3': ['S3_2101', 'S3_2202', 'S3_2303'],
}
DATASETS = ['val', 'test']
CONFIGS = ['M3', 'C025', 'B025']
RHOS = [0.005, 0.01, 0.02, 0.05, 0.10]
PAIR_RHOS = ['rho_0.02', 'rho_prevalence']
PAIRS = ['C-B', 'C-M3', 'B-M3']
N_BOOT = int(os.environ.get('CCMS_NBOOT', '1000'))
BOOT_SEED = 20261101


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_entry(path):
    return {'path': str(path.relative_to(ROOT)),
            'n_bytes': path.stat().st_size,
            'sha256': sha256(path)}


def env_metadata():
    import sklearn
    return {'version': sys.version.split()[0], 'version_full': sys.version,
            'executable': sys.executable, 'platform': sys.platform,
            'numpy': np.__version__, 'sklearn': sklearn.__version__}


def interval(v):
    return [float(z) for z in np.quantile(v, [.025, .975])]


def alert_budget(rho, n):
    """ceil(rho * n), robust to float such as 0.02*9000 = 180.00000000000003"""
    return max(1, min(n, int(math.ceil(round(rho * n, 9)))))


def point_metrics(y, p):
    """decision-level metrics for one (config, split, replica) score vector"""
    n = len(y)
    npos = int(y.sum())
    order = np.argsort(-p, kind='stable')
    yo = y[order].astype(np.int64)
    cp = np.cumsum(yo)
    out = {'n_rows': n, 'n_pos': npos, 'prevalence': npos / n,
           'alerts_at': {}, 'recall_at': {}, 'precision_at': {}}
    for rho in RHOS:
        m = alert_budget(rho, n)
        tp = int(cp[m - 1])
        out['alerts_at'][f'{rho:g}'] = m
        out['recall_at'][f'{rho:g}'] = tp / npos
        out['precision_at'][f'{rho:g}'] = tp / m
    # budget fixed at the split's own prevalence: ceil(prevalence*n) == npos
    m_prev = alert_budget(npos / n, n)
    tp_prev = int(cp[m_prev - 1])
    out['alerts_prevalence'] = m_prev
    out['recall_prevalence'] = tp_prev / npos
    out['precision_prevalence'] = tp_prev / m_prev
    # alert budget needed to reach recall = 0.5
    k = max(1, int(math.ceil(0.5 * npos)))
    pos_rank = np.flatnonzero(yo == 1)
    m50 = int(pos_rank[k - 1]) + 1
    out['alerts_for_recall_0.5'] = m50
    out['budget_fraction_for_recall_0.5'] = m50 / n
    out['precision_at_recall_0.5'] = k / m50
    return out


def mean_over_records(records):
    """recursive element-wise mean of per-replica metric dicts (floats at leaves)"""
    if isinstance(records[0], dict):
        return {k: mean_over_records([r[k] for r in records]) for k in records[0]}
    return float(np.mean(np.asarray(records, dtype=float)))


def split_stats(values):
    """recursive five-split mean / sd of per-split metric dicts"""
    if isinstance(values[0], dict):
        return {k: split_stats([v[k] for v in values]) for k in values[0]}
    arr = np.asarray(values, dtype=float)
    return {'mean': float(arr.mean()), 'sd_across_splits': float(arr.std(ddof=1))}


def recall_of(fam_point, f, si, cfg, rh):
    v = fam_point[(f, si, cfg)]
    return v['recall_prevalence'] if rh == 'rho_prevalence' else v['recall_at']['0.02']


def main():
    t0 = time.perf_counter()
    packs = [np.load(R / f'aps_seed{s}.npz') for s in SEEDS]

    results = {}
    paired = {}

    for dataset in DATASETS:
        ys = [z['y_' + dataset] for z in packs]
        ids = [z['val_rows'] for z in packs] if dataset == 'val' else [np.arange(len(ys[0])) for _ in packs]
        union = np.unique(np.concatenate(ids))
        uy = np.empty(len(union), int)
        mappings = [np.searchsorted(union, v) for v in ids]
        for y, m in zip(ys, mappings):
            uy[m] = y

        # ---------- point estimates ----------
        results[dataset] = {}
        fam_point = {}
        for f, conds in FAM.items():
            results[dataset][f] = {}
            for cfg in CONFIGS:
                per_split = []
                for si, (y, z) in enumerate(zip(ys, packs)):
                    replica_metrics = [point_metrics(y, z[f'{cfg}__{dataset}__{k}']) for k in conds]
                    fam_point[(f, si, cfg)] = mean_over_records(replica_metrics)
                    per_split.append({'seed': SEEDS[si], 'n_replicas': len(conds),
                                      **fam_point[(f, si, cfg)]})
                results[dataset][f][cfg] = {
                    'per_split': per_split,
                    'five_split_mean': split_stats(
                        [{k: v for k, v in r.items() if k not in ('seed', 'n_replicas')} for r in per_split]),
                }
            print(f'[A3] {dataset} {f} point metrics done ({time.perf_counter() - t0:.1f}s)', flush=True)

        # ---------- paired bootstrap of recall differences ----------
        cache = {}
        for f, conds in FAM.items():
            for si, (y, z) in enumerate(zip(ys, packs)):
                for ci, k in enumerate(conds):
                    for cfg in CONFIGS:
                        p = z[f'{cfg}__{dataset}__{k}']
                        o = np.argsort(-p, kind='stable')
                        cache[(f, si, ci, cfg)] = (o, y[o].astype(np.int64))

        draws = {(f, pair, rh): np.zeros((N_BOOT, 5)) for f in FAM for pair in PAIRS for rh in PAIR_RHOS}
        rng = np.random.default_rng(BOOT_SEED)          # reset once per dataset, as in aps_boot
        pidx = np.flatnonzero(uy == 1)
        nidx = np.flatnonzero(uy == 0)
        for b in range(N_BOOT):
            ix = np.r_[rng.choice(pidx, len(pidx), replace=True),
                       rng.choice(nidx, len(nidx), replace=True)]
            weights = np.bincount(ix, minlength=len(union)).astype(float)
            for f, conds in FAM.items():
                for si, m in enumerate(mappings):
                    w = weights[m]
                    stats = {}
                    for ci in range(len(conds)):
                        for cfg in CONFIGS:
                            o, yo = cache[(f, si, ci, cfg)]
                            ww = w[o]
                            stats[(ci, cfg)] = (np.cumsum(ww), np.cumsum(ww * yo))
                    for rh in PAIR_RHOS:
                        if rh == 'rho_0.02':
                            rho = 0.02
                        else:
                            rho = fam_point[(f, si, 'M3')]['prevalence']
                        rec = {cfg: [] for cfg in CONFIGS}
                        for ci in range(len(conds)):
                            for cfg in CONFIGS:
                                cw, cpos = stats[(ci, cfg)]
                                j = int(np.searchsorted(cw, rho * cw[-1], side='left'))
                                if j >= len(cw):
                                    j = len(cw) - 1
                                rec[cfg].append(cpos[j] / cpos[-1])
                        draws[(f, 'C-B', rh)][b, si] = np.mean(
                            [a - bq for a, bq in zip(rec['C025'], rec['B025'])])
                        draws[(f, 'C-M3', rh)][b, si] = np.mean(
                            [a - bq for a, bq in zip(rec['C025'], rec['M3'])])
                        draws[(f, 'B-M3', rh)][b, si] = np.mean(
                            [a - bq for a, bq in zip(rec['B025'], rec['M3'])])
            if b % 250 == 0:
                print(f'[A3] {dataset} boot={b} elapsed={time.perf_counter() - t0:.1f}s', flush=True)

        paired[dataset] = {}
        for f in FAM:
            paired[dataset][f] = {}
            for rh in PAIR_RHOS:
                paired[dataset][f][rh] = {}
                for pair in PAIRS:
                    ca, cb_ = pair.split('-')
                    first = 'C025' if ca == 'C' else 'B025'
                    second = 'M3' if cb_ == 'M3' else 'B025'
                    point_per_split = np.array([
                        recall_of(fam_point, f, si, first, rh) - recall_of(fam_point, f, si, second, rh)
                        for si in range(5)])
                    d = draws[(f, pair, rh)]
                    paired[dataset][f][rh][pair] = {
                        'delta_mean': float(np.mean(point_per_split)),
                        'ci95': interval(d.mean(1)),
                        'per_split': [{'seed': SEEDS[i], 'delta': float(point_per_split[i]),
                                       'ci95': interval(d[:, i])} for i in range(5)],
                    }

        # consistency: paired delta_mean must equal five-split means difference
        cons = 0.0
        for f in FAM:
            for rh in PAIR_RHOS:
                for pair in PAIRS:
                    ca, cb_ = pair.split('-')
                    first = 'C025' if ca == 'C' else 'B025'
                    second = 'M3' if cb_ == 'M3' else 'B025'
                    key = 'recall_at' if rh == 'rho_0.02' else None
                    def five_mean(cfg):
                        m = results[dataset][f][cfg]['five_split_mean']
                        return m['recall_at']['0.02']['mean'] if rh == 'rho_0.02' else m['recall_prevalence']['mean']
                    cons = max(cons, abs(paired[dataset][f][rh][pair]['delta_mean']
                                         - (five_mean(first) - five_mean(second))))
        print(f'[A3] {dataset} paired point-estimate consistency max abs diff {cons:.3e}', flush=True)

    payload = {
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'python': env_metadata(),
        'sources': [source_entry(R / f'aps_seed{s}.npz') for s in SEEDS]
                   + [source_entry(PKG / 'scripts' / 'summarize.py')],
        'protocol': {
            'configs': {'M3': 'reference trunk (median imputation + all missing flags + HGB)', 'C025': 'class-conditional shrink at alpha=0.25', 'B025': 'class-blind (marginal) shrink at alpha=0.25 with matched expected deletion budget'},
            'families': FAM,
            'datasets': DATASETS,
            'families_replicated': 'inside a family the three stress replicas are scored separately and every metric is averaged over the replicas afterwards; scores are never averaged',
            'alert_rule': 'rank rows by predicted score descending (np.argsort(-p, kind="stable")); alert the top ceil(rho*N) rows; recall = TP/n_pos; precision = TP/ceil(rho*N)',
            'alert_rates': RHOS,
            'prevalence_budget': 'rho is fixed at the split prevalence P/N, i.e. the budget is ceil(prevalence*N) = P alerts; at this budget precision equals recall by construction',
            'recall_0.5_budget': 'the rank of the ceil(0.5*n_pos)-th highest-scored positive, divided by N: the smallest alert count that reaches recall = 0.5',
            'five_split_mean': 'per-split family values averaged over the five splits; sd_across_splits uses ddof=1',
            'bootstrap': {
                'n_bootstrap': N_BOOT,
                'seed': BOOT_SEED,
                'replicates': 'summarize.py aps_boot resampling verbatim: val = np.unique union of the five val_rows with np.searchsorted mapping and one shared weight vector per replicate (weights shared across the five splits); test = np.arange(len(y_test)) IDs; per replicate pos then neg rng.choice; rng reset once per dataset',
                'paired_estimand': 'per split, per family: mean over the three replicas of (recall(config A) - recall(config B)) at the same alert rate; family draw = mean over the five splits; ci95 = 2.5/97.5 percentile of the 1000 draws',
                'weighted_recall_at_budget': 'rows in the fixed stable score order; cumulative resample weight cw and cumulative positive weight cpos; take the first row with cw >= rho*cw[-1]; reproduces the plain top-ceil(rho*N) rule under unit weights',
                'prevalence_rate_used_in_bootstrap': 'rho = the split prevalence of the point estimate (constant across replicates), not the resampled prevalence',
            },
            'pairs': {'C-B': 'primary (matches the R13/R12 headline contrast)', 'C-M3': 'supplementary', 'B-M3': 'supplementary'},
            'sign_convention': 'positive delta = first config has higher recall',
        },
        'results': results,
        'paired_recall_differences': paired,
    }
    (OUT / 'r15_decision_metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    # ---------------- console report ----------------
    print('\n[A3] recall at alert budget rho (five-split mean, family mean over replicas)')
    print(f"  {'':14s}" + ''.join(f"{r:>10s}" for r in [f'{x:g}' for x in RHOS]) + f"{'prev':>10s}")
    for dataset in DATASETS:
        for f in FAM:
            for cfg in CONFIGS:
                m = results[dataset][f][cfg]['five_split_mean']
                vals = ''.join(f"{m['recall_at'][f'{r:g}']['mean']:10.4f}" for r in RHOS)
                print(f'  {dataset:4s} {f:6s} {cfg:5s}' + vals
                      + f"{m['recall_prevalence']['mean']:10.4f}")
    print('\n[A3] recall=0.5 required alert fraction (five-split mean)')
    for dataset in DATASETS:
        for f in FAM:
            row = [f"{cfg}={results[dataset][f][cfg]['five_split_mean']['budget_fraction_for_recall_0.5']['mean']:.4f}"
                   for cfg in CONFIGS]
            print(f'  {dataset:4s} {f:6s} ' + ' '.join(row))
    for rh in PAIR_RHOS:
        print(f'\n[A3] paired recall differences at {rh} (positive = first higher)')
        for dataset in DATASETS:
            for f in FAM:
                cells = []
                for pair in PAIRS:
                    p = paired[dataset][f][rh][pair]
                    cells.append(f"{pair}: {p['delta_mean']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]")
                print(f'  {dataset:4s} {f:6s} ' + ' | '.join(cells))
    print(f"\n[A3] wrote {OUT / 'r15_decision_metrics.json'} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
