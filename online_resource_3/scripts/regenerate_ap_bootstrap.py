#!/usr/bin/env python
"""Portable adapter for the frozen APS 20,000-draw generator.

Uses packaged saved predictions and the historical first-1,000-draw references.
CCMS_NBOOT defaults to 20000. This program does not retrain any model.
Historical ratio field names are retained for machine compatibility and denote
algebraic path descriptions, not pure mechanisms or causal attribution.
See CHANGELOG.md for the path-only adapter changes and inferential boundaries.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "aps_predictions"
OUT = Path(os.environ.get("CCMS_OUT", str(ROOT / "outputs" / "ap_bootstrap20k"))).resolve()
OUT.mkdir(parents=True, exist_ok=False)

SEEDS = [11, 22, 33, 44, 55]
FAM = {
    'S0': ['S0'],
    'S1_30': ['S1_30_101', 'S1_30_202', 'S1_30_303'],
    'S2': ['S2_1101', 'S2_1202', 'S2_1303'],
    'S3': ['S3_2101', 'S3_2202', 'S3_2303'],
}
DATASETS = ['val', 'test']
PAIRS = ['C-B', 'C-M3', 'B-M3']
N_BOOT = int(os.environ.get('CCMS_NBOOT', '20000'))
BOOT_SEED = 20261101
TOL = 1e-6
DRAW_TOL = 1e-12


# --------------------------------------------------------------------------
# the four helpers below are copied verbatim from scripts/summarize.py
# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_entry(path):
    return {
        'path': str(path.relative_to(ROOT)),
        'n_bytes': path.stat().st_size,
        'sha256': sha256(path),
    }


def env_metadata():
    import sklearn
    return {
        'version': sys.version.split()[0],
        'version_full': sys.version,
        'executable': sys.executable,
        'platform': sys.platform,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
    }


def ratio(num, den, digits=None):
    """full-precision ratio, None when the denominator is exactly 0"""
    if den == 0.0 or not np.isfinite(den):
        return None
    v = float(num / den)
    return v


def decompose(m):
    """split C-B into the C-over-M3 total and the B-over-M3 (baseline) part.

    identity:  C-B = (C-M3) - (B-M3)
    """
    cm3, bm3, cb = m['C-M3'], m['B-M3'], m['C-B']
    return {
        'c_minus_b': cb,
        'c_minus_m3': cm3,
        'b_minus_m3': bm3,
        'identity_residual_cb_minus_cm3_plus_bm3': float(cb - (cm3 - bm3)),
        'amplification_c_minus_b_over_c_minus_m3': ratio(cb, cm3),
        'b_minus_m3_share_of_c_minus_m3_total_gain': ratio(bm3, cm3),
        'c_minus_b_share_of_c_minus_m3_total_gain': ratio(cb, cm3),
        'b_minus_m3_over_c_minus_b': ratio(bm3, cb),
        'abs_b_minus_m3_over_abs_c_minus_b': ratio(abs(bm3), abs(cb)),
        'sign_agreement_c_minus_b_vs_c_minus_m3': (cb < 0) == (cm3 < 0),
    }


def main():
    t0 = time.perf_counter()
    packs = [np.load(R / f'aps_seed{s}.npz') for s in SEEDS]

    ref_json_path = R / 'aps_matched_bootstrap.json'
    ref_draws_path = R / 'aps_bootstrap_draws.npz'
    ref = json.loads(ref_json_path.read_text())
    ref_draws = np.load(ref_draws_path)

    results = {}
    draws_out = {}
    recon_cells = []
    recon_max = {'delta_mean': 0.0, 'ci95': 0.0, 'per_split_delta': 0.0,
                 'per_split_ci95': 0.0, 'draws': 0.0}

    for dataset in DATASETS:
        ys = [z['y_' + dataset] for z in packs]
        ids = [z['val_rows'] for z in packs] if dataset == 'val' else [np.arange(len(ys[0])) for _ in packs]
        union = np.unique(np.concatenate(ids))
        uy = np.empty(len(union), int)
        mappings = [np.searchsorted(union, v) for v in ids]
        for y, m in zip(ys, mappings):
            uy[m] = y

        # ---- point estimates (unweighted sklearn AP), prepped score matrices ----
        pr = {}
        aps = {}
        points = {f: {p: np.zeros(5) for p in PAIRS} for f in FAM}
        for f, conds in FAM.items():
            pr[f] = []
            for si, (y, z) in enumerate(zip(ys, packs)):
                row = []
                for k in conds:
                    aC = float(AP(y, z['C025__' + dataset + '__' + k]))
                    aB = float(AP(y, z['B025__' + dataset + '__' + k]))
                    aM = float(AP(y, z['M3__' + dataset + '__' + k]))
                    aps[(f, si, k)] = (aC, aB, aM)
                    row.append((prep(y, z['C025__' + dataset + '__' + k]),
                                prep(y, z['B025__' + dataset + '__' + k]),
                                prep(y, z['M3__' + dataset + '__' + k])))
                pr[f].append(row)
            for pair in PAIRS:
                if pair == 'C-B':
                    points[f][pair] = np.array([float(np.mean([aps[(f, si, k)][0] - aps[(f, si, k)][1] for k in conds])) for si in range(5)])
                elif pair == 'C-M3':
                    points[f][pair] = np.array([float(np.mean([aps[(f, si, k)][0] - aps[(f, si, k)][2] for k in conds])) for si in range(5)])
                else:
                    points[f][pair] = np.array([float(np.mean([aps[(f, si, k)][1] - aps[(f, si, k)][2] for k in conds])) for si in range(5)])

        # ---- paired bootstrap, identical RNG stream to aps_boot ----
        boots = {f: {p: np.zeros((N_BOOT, 5)) for p in PAIRS} for f in FAM}
        rng = np.random.default_rng(BOOT_SEED)          # reset once per dataset
        pidx = np.flatnonzero(uy == 1)
        nidx = np.flatnonzero(uy == 0)
        for b in range(N_BOOT):
            ix = np.r_[rng.choice(pidx, len(pidx), replace=True),
                       rng.choice(nidx, len(nidx), replace=True)]
            weights = np.bincount(ix, minlength=len(union)).astype(float)
            for f in FAM:
                for si, m in enumerate(mappings):
                    w = weights[m]
                    dcb, dcm, dbm = [], [], []
                    for (pc, pb, pm) in pr[f][si]:
                        wc = wap(pc, w)
                        wb = wap(pb, w)
                        wm = wap(pm, w)
                        dcb.append(wc - wb)
                        dcm.append(wc - wm)
                        dbm.append(wb - wm)
                    boots[f]['C-B'][b, si] = np.mean(dcb)
                    boots[f]['C-M3'][b, si] = np.mean(dcm)
                    boots[f]['B-M3'][b, si] = np.mean(dbm)
            if b % 250 == 0:
                print(f'[A1] dataset={dataset} boot={b} elapsed={time.perf_counter() - t0:.1f}s', flush=True)

        # ---- aggregate + reconcile ----
        results[dataset] = {}
        for f in FAM:
            cell = {}
            for pair in PAIRS:
                cell[pair] = {
                    'delta_mean': float(np.mean(points[f][pair])),
                    'ci95': interval(boots[f][pair].mean(1)),
                    'per_split': [
                        {'seed': s, 'delta': float(points[f][pair][i]),
                         'ci95': interval(boots[f][pair][:, i])}
                        for i, s in enumerate(SEEDS)
                    ],
                }
                draws_out[f'{dataset}__{f}__{pair}'] = boots[f][pair]
            cell['decomposition'] = decompose({p: cell[p]['delta_mean'] for p in PAIRS})
            cell['decomposition']['per_split'] = [
                dict(seed=SEEDS[i],
                     **decompose({p: float(points[f][p][i]) for p in PAIRS}))
                for i in range(5)
            ]
            results[dataset][f] = cell

            ref_cell = ref['results'][dataset][f]
            my_cell = cell['C-B']
            dd = abs(my_cell['delta_mean'] - ref_cell['delta_mean'])
            dc = max(abs(my_cell['ci95'][0] - ref_cell['ci95'][0]),
                     abs(my_cell['ci95'][1] - ref_cell['ci95'][1]))
            dpsd = max(abs(s['delta'] - r['delta']) for s, r in zip(my_cell['per_split'], ref_cell['per_split']))
            dpsc = max(max(abs(s['ci95'][j] - r['ci95'][j]) for j in (0, 1))
                       for s, r in zip(my_cell['per_split'], ref_cell['per_split']))
            ref_boot = ref_draws[f'{dataset}__{f}']
            nb = min(N_BOOT, ref_boot.shape[0])
            ddr = float(np.max(np.abs(boots[f]['C-B'][:nb] - ref_boot[:nb])))
            if N_BOOT != 1000:
                print(f'[A1] EXTENDED MODE ({N_BOOT} draws): first {nb} draws checked against archived stream; new CI estimated from all draws', flush=True)
            recon_cells.append({
                'dataset': dataset, 'family': f,
                'delta_mean': my_cell['delta_mean'], 'reference_delta_mean': ref_cell['delta_mean'],
                'abs_diff_delta_mean': dd, 'ci95': my_cell['ci95'],
                'reference_ci95': ref_cell['ci95'], 'abs_diff_ci95_max': dc,
                'abs_diff_per_split_delta_max': dpsd, 'abs_diff_per_split_ci95_max': dpsc,
                'draws_abs_diff_max': ddr,
                'pass': bool(dd < TOL and dc < TOL and dpsd < TOL and dpsc < TOL and ddr < DRAW_TOL),
            })
            recon_max['delta_mean'] = max(recon_max['delta_mean'], dd)
            recon_max['ci95'] = max(recon_max['ci95'], dc)
            recon_max['per_split_delta'] = max(recon_max['per_split_delta'], dpsd)
            recon_max['per_split_ci95'] = max(recon_max['per_split_ci95'], dpsc)
            recon_max['draws'] = max(recon_max['draws'], ddr)

    if N_BOOT != 1000:
        for c in recon_cells:
            c['pass'] = bool(c['abs_diff_delta_mean'] < TOL and c['draws_abs_diff_max'] < DRAW_TOL)
            c['pass_note'] = 'Point estimates and archived first 1,000 draws match; the 20,000-draw interval is intentionally re-estimated.'
    passed = (all(c['pass'] for c in recon_cells) and len(recon_cells) == 8)

    # identity residuals across all cells (should be ~1e-17)
    id_max = 0.0
    for dataset in DATASETS:
        for f in FAM:
            id_max = max(id_max, abs(results[dataset][f]['decomposition']['identity_residual_cb_minus_cm3_plus_bm3']))
            for sp in results[dataset][f]['decomposition']['per_split']:
                id_max = max(id_max, abs(sp['identity_residual_cb_minus_cm3_plus_bm3']))

    recon = {
        'hard_check': 'recomputed C025-B025 vs results/aps_matched_bootstrap.json and results/aps_bootstrap_draws.npz',
        'reference_json': source_entry(ref_json_path),
        'reference_draws': source_entry(ref_draws_path),
        'tolerance_delta_and_ci': TOL,
        'tolerance_draws': DRAW_TOL,
        'extended_bootstrap_note': ('For N_BOOT>1000, the hard check verifies point estimates and the archived first 1,000 draws; intervals are intentionally re-estimated from all draws.' if N_BOOT != 1000 else 'All point estimates, intervals and draws are checked against the archive.'),
        'n_cells_checked': len(recon_cells),
        'max_abs_diff': recon_max,
        'identity_C-B=(C-M3)-(B-M3)_max_abs_residual': float(id_max),
        'passed': bool(passed),
        'per_cell': recon_cells,
    }

    print('\n[A1] reconciliation vs aps_matched_bootstrap.json', flush=True)
    for c in recon_cells:
        print(f"  {c['dataset']:4s} {c['family']:6s} d_mean={c['delta_mean']:.10f} "
              f"ref={c['reference_delta_mean']:.10f} |dd|={c['abs_diff_delta_mean']:.2e} "
              f"|dci|={c['abs_diff_ci95_max']:.2e} |ddraw|={c['draws_abs_diff_max']:.2e} "
              f"{'PASS' if c['pass'] else 'FAIL'}", flush=True)
    print(f"[A1] MAX delta_mean {recon_max['delta_mean']:.3e} | ci95 {recon_max['ci95']:.3e} | "
          f"per_split delta {recon_max['per_split_delta']:.3e} | per_split ci95 {recon_max['per_split_ci95']:.3e} | "
          f"draws {recon_max['draws']:.3e}", flush=True)

    if not passed:
        print('\n[A1] ARCHIVED POINT/STREAM CHECK FAILED -- nothing written.', flush=True)
        return 2

    payload = {
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'python': env_metadata(),
        'sources': [source_entry(R / f'aps_seed{s}.npz') for s in SEEDS]
                   + [source_entry(ROOT / 'frozen' / 'legacy_ap' / 'summarize.py')],
        'reconciliation': recon,
        'protocol': {
            'replicates': 'summarize.py aps_boot() prep/wap/interval copied verbatim; M3 added inside the same replicate loop without consuming extra RNG',
            'n_bootstrap': N_BOOT,
            'bootstrap_seed': BOOT_SEED,
            'rng_reset': 'np.random.default_rng(20261101) reset once per dataset (val, test) exactly as in aps_boot',
            'val_sampling': 'unique original validation IDs (np.unique of the five val_rows) stratified by label; one shared weight vector per replicate, sliced by np.searchsorted(union, val_rows) mapping; weights shared across the five splits',
            'test_sampling': 'np.arange(len(y_test)) as IDs; the same official test row weights are shared across all five fitted models and all stress inputs',
            'per_replicate_draw': 'ix = r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]; weights = np.bincount(ix, minlength=len(union)).astype(float)',
            'estimator': 'per split, per stress family: mean over the 3 family replicas of AP(A) - AP(B) (unweighted sklearn AP point estimate; weighted-AP difference inside the bootstrap); family estimate = mean over the five splits; ci95 = np.quantile(draw_mean_over_splits, [.025, .975])',
            'point_estimand_note': 'point estimate uses sklearn.metrics.average_precision_score on raw scores for each replica, then averages the per-replica differences (identical to summarize.py); the bootstrap uses the same weighted-AP wap() on the resampled weights',
            'configs': {'C': 'C025 (alpha 0.25 conditional/class-conditional shrink)', 'B': 'B025 (alpha 0.25 marginal/base shrink)', 'M3': 'Stored unmodified single BASE model; not a nonzero ens3 candidate'},
            'families': FAM,
            'datasets': DATASETS,
            'pairs': PAIRS,
            'sign_convention': 'delta = AP(first) - AP(second); negative delta means C025/B025 is WORSE than the comparator',
        },
        'results': results,
    }
    (OUT / 'r13_threeway_20k.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    np.savez_compressed(OUT / 'aps_threeway_draws_20k.npz', **draws_out)

    print(f'\n[A1] wrote {OUT / "r13_threeway_20k.json"} and {OUT / "aps_threeway_draws_20k.npz"} '
          f'({time.perf_counter() - t0:.1f}s)', flush=True)

    # compact console table for the report
    for dataset in DATASETS:
        for f in FAM:
            d = results[dataset][f]['decomposition']
            print(f"[A1] {dataset:4s} {f:6s} C-B={d['c_minus_b']:+.4f} C-M3={d['c_minus_m3']:+.4f} "
                  f"B-M3={d['b_minus_m3']:+.4f} amp={d['amplification_c_minus_b_over_c_minus_m3']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
