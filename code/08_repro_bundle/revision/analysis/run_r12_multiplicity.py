#!/usr/bin/env python
"""A2 / R12: multiplicity correction for the EIGHT nominal 95% C-B intervals
(val/test x S0, S1_30, S2, S3) treated as ONE simultaneous family.

Inputs (produced by run_r13_threeway.py, same sampling protocol):
  revision/analysis/r13_threeway.json          -> point estimates, nominal CIs
  revision/analysis/aps_threeway_draws.npz     -> C-B bootstrap draws (1000, 5)

Methods (as specified in the R12 task card):
  1. Bonferroni   : equal-tailed percentile interval of the 1000 family-level
                    draws at the adjusted two-sided level 0.05/8
                    -> quantiles (0.025/8, 1-0.025/8).
  2. Holm         : step-down on bootstrap two-sided p-values, m = 8.
                    Card formula  p* = (1 + #{|delta*_b| >= |delta_hat|}) / 1001
                    is computed verbatim as p_holm_literal, BUT it is reported
                    as degenerate: the draws are centred on delta_hat, so
                    |delta*_b| >= |delta_hat| holds for ~50% of draws for every
                    cell and no cell can ever be rejected.  The statistically
                    valid two-sided bootstrap p-value (same estimator, null
                    recentred) is therefore ALSO computed:
                    p = (1 + #{|delta*_b - delta_hat| >= |delta_hat|}) / 1001
                    Holm is applied to both and both decisions are reported.
  3. sup-t        : t*_b = (delta*_b - delta_hat)/sd_b, c = 97.5% quantile of
                    max over the 8 cells of |t*_b|, band = delta_hat +- c*sd_b.

Output: revision/analysis/r12_multiplicity.json
Re-run:  python run_r12_multiplicity.py   (requires A1 outputs)
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # bundle root
PKG = ROOT / 'CCMS论文修订完整资料包_2026-11-01_192149' / '02_本轮代码与结果'
R = PKG / 'results'
OUT = ROOT / 'revision' / 'analysis'

FAMILIES = ['S0', 'S1_30', 'S2', 'S3']
DATASETS = ['val', 'test']
CELLS = [(d, f) for d in DATASETS for f in FAMILIES]
M = len(CELLS)              # 8 simultaneous cells
N_BOOT = 1000
ALPHA = 0.05
FWER = 0.05


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


def holm(pvals, m, alpha=ALPHA):
    """step-down Holm; returns adjusted p, ascending rank, per-step alpha threshold,
    and the margin to that threshold (negative = reject) for every cell"""
    order = np.argsort(pvals, kind='stable')
    adj = np.empty(m)
    rank = np.empty(m, int)          # 1-based rank in ascending p order
    thr = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * pvals[j])
        adj[j] = min(1.0, running)
        rank[j] = i + 1
        thr[j] = alpha / (m - (i + 1) + 1)      # alpha/(m-i) with i 1-based
    return adj, rank, thr, pvals - thr


def main():
    r13_path = OUT / 'r13_threeway.json'
    draws_path = OUT / 'aps_threeway_draws.npz'
    r13 = json.loads(r13_path.read_text())
    draws = np.load(draws_path)

    rows = []
    for dataset, fam in CELLS:
        cell = r13['results'][dataset][fam]['C-B']
        d_hat = float(cell['delta_mean'])
        d = draws[f'{dataset}__{fam}__C-B'].mean(1)          # (1000,) family-level draws
        sd = float(np.std(d, ddof=1))
        rows.append({
            'dataset': dataset, 'family': fam, 'cell': f'{dataset}__{fam}',
            'delta_hat': d_hat,
            'sd_boot': sd,
            'nominal_ci95': [float(x) for x in cell['ci95']],
            'nominal_excludes_zero': bool(cell['ci95'][0] > 0 or cell['ci95'][1] < 0),
            'n_draws': int(d.shape[0]),
            # 1. Bonferroni percentile
            'bonferroni': {
                'level_two_sided': ALPHA / M,
                'quantiles': [0.025 / M, 1 - 0.025 / M],
                'ci': [float(np.quantile(d, 0.025 / M)), float(np.quantile(d, 1 - 0.025 / M))],
            },
            # 2. Holm inputs
            'p_two_sided_literal': float((1 + np.sum(np.abs(d) >= abs(d_hat))) / (N_BOOT + 1)),
            'p_two_sided_recentered': float((1 + np.sum(np.abs(d - d_hat) >= abs(d_hat))) / (N_BOOT + 1)),
            'p_one_sided_percentile': float((1 + np.sum(d >= 0)) / (N_BOOT + 1)),
            # 3. sup-t inputs
            't_star': (d - d_hat) / sd,
        })

    # ---- 1. Bonferroni verdicts ----
    for r in rows:
        lo, hi = r['bonferroni']['ci']
        r['bonferroni']['excludes_zero'] = bool(lo > 0 or hi < 0)
        r['bonferroni']['significant_at_fwer_0.05'] = r['bonferroni']['excludes_zero']

    # ---- 2. Holm on both p-value variants ----
    for key, tag in [('p_two_sided_recentered', 'holm_recentered'),
                     ('p_two_sided_literal', 'holm_literal')]:
        p = np.array([r[key] for r in rows])
        adj, rank, thr, margin = holm(p, M)
        for r, a, rk, t, mg in zip(rows, adj, rank, thr, margin):
            r[tag] = {'p_raw': float(r[key]),
                      'p_holm_adjusted': float(a),
                      'rank_ascending': int(rk),
                      'step_alpha_threshold': float(t),
                      'margin_p_minus_threshold': float(mg),
                      'margin_in_bootstrap_draws': float(mg * (N_BOOT + 1)),
                      'reject_at_fwer_0.05': bool(a <= FWER)}

    # ---- 3. sup-t simultaneous band ----
    T = np.vstack([r['t_star'] for r in rows])               # (8, 1000)
    max_abs_t = np.max(np.abs(T), axis=0)                    # (1000,)
    c = float(np.quantile(max_abs_t, 1 - ALPHA / 2))
    for r in rows:
        lo = r['delta_hat'] - c * r['sd_boot']
        hi = r['delta_hat'] + c * r['sd_boot']
        r['sup_t'] = {'c_975': c, 'ci': [float(lo), float(hi)],
                      'excludes_zero': bool(lo > 0 or hi < 0),
                      'significant_at_fwer_0.05': bool(lo > 0 or hi < 0)}
        del r['t_star']

    # ---- S0 specific check demanded by R12 ----
    s0 = {}
    for dataset in DATASETS:
        r = next(x for x in rows if x['dataset'] == dataset and x['family'] == 'S0')
        hr = r['holm_recentered']
        s0[dataset] = {
            'delta_hat': r['delta_hat'],
            'nominal_ci95': r['nominal_ci95'],
            'nominal_excludes_zero': r['nominal_excludes_zero'],
            'nominal_margin_of_upper_bound_to_zero': float(r['nominal_ci95'][1]),   # = -0.000806 for val
            'bonferroni_ci': r['bonferroni']['ci'],
            'bonferroni_excludes_zero': r['bonferroni']['excludes_zero'],
            'bonferroni_upper_bound_above_zero_by': float(r['bonferroni']['ci'][1]),
            'sup_t_ci': r['sup_t']['ci'],
            'sup_t_excludes_zero': r['sup_t']['excludes_zero'],
            'p_two_sided_recentered': r['p_two_sided_recentered'],
            'n_draws_supporting_recentered': int(round(r['p_two_sided_recentered'] * (N_BOOT + 1))) - 1,
            'holm_recentered_adjusted': hr['p_holm_adjusted'],
            'holm_recentered_step_alpha_threshold': hr['step_alpha_threshold'],
            'holm_recentered_margin_in_draws': hr['margin_in_bootstrap_draws'],
            'holm_recentered_reject': hr['reject_at_fwer_0.05'],
            'holm_decision_flips_with_one_extra_draw': bool(abs(hr['margin_in_bootstrap_draws']) <= 1),
        }
    s0_verdict = {
        'simultaneous_interval_methods_exclude_zero': {
            d: bool(s0[d]['bonferroni_excludes_zero'] or s0[d]['sup_t_excludes_zero']) for d in DATASETS},
        'note': ('Both simultaneous-interval constructions (Bonferroni-adjusted percentile and sup-t) cover 0 for '
                 'val__S0 and test__S0, so the nominal -0.0008 / -0.0011 boundary bounds are NOT multiplicity-robust. '
                 'The recentred Holm test does reject at FWER 0.05, but only by less than one bootstrap draw '
                 '(margin in draws stored per cell), i.e. it is a knife-edge artefact of the 1000-replicate grid, '
                 'not evidence for a direction claim. Reported conclusion for R12: "8/8 point estimates negative" is '
                 'descriptive only; the S0 direction is not established under simultaneous inference.'),
    }

    any_sig = [f"{r['dataset']}__{r['family']}" for r in rows
               if r['bonferroni']['significant_at_fwer_0.05'] and r['sup_t']['significant_at_fwer_0.05']]
    both_agree = all(r['bonferroni']['significant_at_fwer_0.05'] == r['sup_t']['significant_at_fwer_0.05'] for r in rows)

    payload = {
        'generated_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'python': env_metadata(),
        'sources': [source_entry(r13_path), source_entry(draws_path),
                    source_entry(R / 'aps_matched_bootstrap.json')],
        'depends_on': 'run_r13_threeway.py (A1); draws are the C-B pair of aps_threeway_draws.npz, which are bit-identical to results/aps_bootstrap_draws.npz',
        'protocol': {
            'family_of_tests': M,
            'cells': [f'{d}__{f}' for d, f in CELLS],
            'cell_order': [f'{d}__{f}' for d, f in CELLS],
            'n_bootstrap': N_BOOT,
            'bootstrap_seed': 20261101,
            'draw_definition': 'family-level draw = mean over the five splits of (mean over the three family replicas of weighted-AP(C025) - weighted-AP(B025)); identical to the A1 estimator',
            'sd_definition': 'sample sd (ddof=1) of the 1000 family-level draws',
            'alpha': ALPHA,
            'fwer': FWER,
            'bonferroni': 'equal-tailed percentile interval of the 1000 draws evaluated at (0.025/8, 1-0.025/8)',
            'holm': {
                'm': M,
                'two_sided_p_literal_card_formula': '(1 + #{|delta*_b| >= |delta_hat|}) / 1001  [computed verbatim as requested]',
                'two_sided_p_recentered_used_for_inference': '(1 + #{|delta*_b - delta_hat| >= |delta_hat|}) / 1001',
                'degeneracy_note': 'The card formula is degenerate here: the bootstrap draws are centred on delta_hat, so |delta*_b| >= |delta_hat| holds for about half of the draws in EVERY cell (p ~ 0.5) and Holm can never reject. The recentred form is the standard two-sided bootstrap p-value and is the one used for the Holm decision reported as primary; the literal variant is kept in the JSON for audit.',
                'monotonicity': 'adjusted p = max over j<=i of (m-j+1)*p_(j), clipped to 1',
            },
            'sup_t': 't*_b = (delta*_b - delta_hat)/sd_b; c = 97.5% quantile of max over the 8 cells of |t*_b| (same replicate index b across cells); band = delta_hat +- c*sd_b. val and test cells share the replicate index although they come from separate RNG streams (rng reset per dataset in A1), so the band is the requested construction rather than a strictly joint resampling of all 8 cells.',
            'nominal': 'nominal 95% intervals are the A1 / aps_matched_bootstrap.json percentile intervals, reproduced bit-exactly',
        },
        'summary': {
            'n_cells': M,
            'all_point_estimates_negative': all(r['delta_hat'] < 0 for r in rows),
            'n_point_estimates_negative': int(sum(r['delta_hat'] < 0 for r in rows)),
            'bonferroni_significant': [f"{r['dataset']}__{r['family']}" for r in rows if r['bonferroni']['significant_at_fwer_0.05']],
            'sup_t_significant': [f"{r['dataset']}__{r['family']}" for r in rows if r['sup_t']['significant_at_fwer_0.05']],
            'holm_recentered_significant': [f"{r['dataset']}__{r['family']}" for r in rows if r['holm_recentered']['reject_at_fwer_0.05']],
            'holm_literal_significant': [f"{r['dataset']}__{r['family']}" for r in rows if r['holm_literal']['reject_at_fwer_0.05']],
            'bonferroni_and_sup_t_agree': bool(both_agree),
            'bonferroni_no_longer_significant_vs_nominal': [
                f"{r['dataset']}__{r['family']}" for r in rows
                if r['nominal_excludes_zero'] and not r['bonferroni']['significant_at_fwer_0.05']],
            'sup_t_c_975': c,
        },
        's0_boundary_check': s0,
        's0_verdict': s0_verdict,
        'cells': rows,
    }
    (OUT / 'r12_multiplicity.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    hdr = f"{'cell':16s} {'delta_hat':>10s} {'nominal ci95':>26s} {'Bonferroni ci':>26s} {'sup-t ci':>26s} {'Holm(rec)':>10s}"
    print(hdr)
    for r in rows:
        print(f"{r['dataset'] + '__' + r['family']:16s} {r['delta_hat']:10.5f} "
              f"[{r['nominal_ci95'][0]:8.5f},{r['nominal_ci95'][1]:8.5f}] "
              f"[{r['bonferroni']['ci'][0]:8.5f},{r['bonferroni']['ci'][1]:8.5f}] "
              f"[{r['sup_t']['ci'][0]:8.5f},{r['sup_t']['ci'][1]:8.5f}] "
              f"{r['holm_recentered']['p_holm_adjusted']:10.4f}")
    print('\nBonferroni significant:', payload['summary']['bonferroni_significant'])
    print('sup-t significant     :', payload['summary']['sup_t_significant'])
    print('Holm recentered signif:', payload['summary']['holm_recentered_significant'])
    print('Holm literal signif   :', payload['summary']['holm_literal_significant'])
    print('S0 val  :', json.dumps(s0['val'], ensure_ascii=False))
    print('S0 test :', json.dumps(s0['test'], ensure_ascii=False))
    print(f'wrote {OUT / "r12_multiplicity.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
