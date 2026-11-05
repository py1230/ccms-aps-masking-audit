#!/usr/bin/env python
"""R6 / B2: seed-level uncertainty and the full nine-multiplier table for the continuous simulation (simulation_continuous.json).

Input: results/simulation_continuous.json (five data seeds x nine target multipliers x all configurations).
  - AP on the official test set for five data seeds (501-505) x nine target multipliers x all configurations.
  - Within a data seed, each configuration averages the predicted probabilities of three training-masking seeds (9107/9108/9109, ens3);
    the seed-level SD therefore reflects variation in the data-generating seed only, not in the masking seed.

Output: revision/analysis/r6_simulation_uncertainty.json
No model is refitted.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2] / "CCMS论文修订完整资料包_2026-11-01_192149"
SIM = PKG / "02_本轮代码与结果/results/simulation_continuous.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = ["M3", "C025", "B025", "C100", "B100"]  # the five columns of the manuscript table
EXTRA = ["M4", "C050", "B050", "C_selected", "B_selected"]
PAIRS = [("C025", "B025"), ("C025", "M3"), ("B025", "M3"), ("C100", "M3"), ("C050", "B050")]
T975_DF4 = 2.7764451051977987  # t_{0.975,4}
N_BOOT = 20000
BOOT_SEED = 20261101


def gkey(g: float) -> str:
    """Result keys as in the archived script: integer multipliers are written as 1 / 0 / -1, the others as 0.75 / -0.25."""
    return f"gap_{g:g}" if g != 0 else "gap_0"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rnd(x, n=6):
    return float(np.round(float(x), n))


def desc(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    return {
        "values_by_data_seed": [rnd(z) for z in v],
        "n": int(len(v)),
        "mean": rnd(v.mean()),
        "sd_seed_level_ddof1": rnd(v.std(ddof=1)),
        "se_of_mean": rnd(v.std(ddof=1) / np.sqrt(len(v))),
        "min": rnd(v.min()),
        "max": rnd(v.max()),
        "range": rnd(v.max() - v.min()),
    }


def t_interval(v: np.ndarray) -> list[float]:
    """t interval over five seeds (df = 4); small sample, used only as a reference scale."""
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return [rnd(v.mean() - T975_DF4 * se), rnd(v.mean() + T975_DF4 * se)]


def boot_interval_mean(v: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Percentile bootstrap over the five seed means (resampling seeds, not rows)."""
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return [rnd(np.quantile(means, 0.025)), rnd(np.quantile(means, 0.975))]


def diff_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    ci = t_interval(d)
    bci = boot_interval_mean(d, rng)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {
        **desc(d),
        "t95_interval_mean_diff": ci,
        "bootstrap95_interval_mean_diff": bci,
        "t_stat_vs_0": rnd(t),
        "n_positive_values": int((d > 0).sum()),
        "n_negative_values": int((d < 0).sum()),
        "all_five_same_sign": bool((d > 0).all() or (d < 0).all()),
        "abs_mean_gt_sd_seed_level": bool(abs(d.mean()) > d.std(ddof=1)),
        "t95_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "bootstrap95_excludes_zero": bool(bci[0] > 0 or bci[1] < 0),
    }


def main():
    t0 = _dt.datetime.now().astimezone()
    d = json.loads(SIM.read_text())
    gaps = [float(g) for g in d["target_gap_multipliers"]]
    seeds = [int(s["data_seed"]) for s in d["seeds"]]
    assert seeds == [501, 502, 503, 504, 505], seeds
    assert len(gaps) == 9, gaps

    configs = [k for k in d["seeds"][0]["metrics"].keys()]
    assert all(c in configs for c in PRIMARY + EXTRA), configs

    def mat(config: str) -> np.ndarray:
        """shape (n_gap, n_seed): target AP for every multiplier and every data seed."""
        return np.array([[float(s["metrics"][config][gkey(g)]) for s in d["seeds"]] for g in gaps])

    rng = np.random.default_rng(BOOT_SEED)
    M = {c: mat(c) for c in configs}

    per_config = {}
    for c in configs:
        per_config[c] = {}
        for i, g in enumerate(gaps):
            per_config[c][gkey(g)] = desc(M[c][i])

    paired = {}
    for ca, cb in PAIRS:
        name = f"{ca}_minus_{cb}"
        paired[name] = {}
        for i, g in enumerate(gaps):
            paired[name][gkey(g)] = diff_stats(M[ca][i], M[cb][i], rng)

    # full nine-multiplier version of the manuscript table (five configurations x nine multipliers, mean +/- SD)
    table7_full = []
    for i, g in enumerate(gaps):
        row = {"target_gap_multiplier": g}
        for c in PRIMARY:
            row[c] = {"mean": rnd(M[c][i].mean()), "sd": rnd(M[c][i].std(ddof=1)),
                      "min": rnd(M[c][i].min()), "max": rnd(M[c][i].max())}
        dc = M["C025"][i] - M["B025"][i]
        row["C025_minus_B025"] = {"mean": rnd(dc.mean()), "sd": rnd(dc.std(ddof=1)),
                                  "t95": t_interval(dc), "boot95": boot_interval_mean(dc, rng),
                                  "all_five_same_sign": bool((dc > 0).all() or (dc < 0).all()),
                                  "abs_mean_gt_sd": bool(abs(dc.mean()) > dc.std(ddof=1))}
        table7_full.append(row)

    # check of why the manuscript shows five multipliers instead of all nine
    shown = [1.0, 0.5, 0.25, 0.0, -1.0]
    omitted = [g for g in gaps if g not in shown]
    robust_flags = {gkey(g): paired["C025_minus_B025"][gkey(g)] for g in gaps}

    # cross-check that the expected overall missingness rate is preserved (every multiplier records expected and actual)
    preservation = []
    for s in d["seeds"]:
        vals = {g: float(s["target_missing"][gkey(g)]["expected_overall_missing"]) for g in gaps}
        acts = {g: float(s["target_missing"][gkey(g)]["actual_missing"]) for g in gaps}
        preservation.append({
            "data_seed": int(s["data_seed"]),
            "expected_overall_missing_min": rnd(min(vals.values())),
            "expected_overall_missing_max": rnd(max(vals.values())),
            "expected_overall_missing_range": rnd(max(vals.values()) - min(vals.values())),
            "actual_missing_at_gap_1": rnd(acts[1.0]),
            "actual_missing_at_gap_-1": rnd(acts[-1.0]),
            "train_expected_overall_missing": rnd(float(s["train_missing"]["expected_overall_missing"])),
        })

    selection = [{"data_seed": int(s["data_seed"]),
                  "C_selected_config": s["source_selection"]["C"]["chosen"]["config"],
                  "C_selected_alpha": float(s["source_selection"]["C"]["chosen"]["alpha"]),
                  "B_selected_config": s["source_selection"]["B"]["chosen"]["config"],
                  "B_selected_alpha": float(s["source_selection"]["B"]["chosen"]["alpha"])}
                 for s in d["seeds"]]

    payload = {
        "generated_at": t0.isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_bootstrap_of_seeds": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "t_quantile_975_df4": T975_DF4,
        "description": (
            "R6/B2: seed-level uncertainty of the continuous simulation over five data seeds x nine target multipliers; within a seed each configuration"
            "averages the three training-masking seeds (ens3), so the SD reflects data-seed variation only. Paired differences are formed within a data seed and multiplier."
        ),
        "sources": [{"path": str(SIM), "sha256": sha256(SIM), "role": "official-test AP for five data seeds x nine multipliers x all configurations"}],
        "data_seeds": seeds,
        "target_gap_multipliers": gaps,
        "configs_paper_table7_columns": PRIMARY,
        "configs_extra": EXTRA,
        "per_config_per_gap": per_config,
        "paired_differences": paired,
        "table7_full_9_multipliers": table7_full,
        "multiplier_robustness_C025_vs_B025": robust_flags,
        "table7_key_multiplier_omission_check": {
            "shown_in_current_manuscript": shown,
            "omitted": omitted,
            "omitted_have_abs_mean_gt_sd": {gkey(g): robust_flags[gkey(g)]["abs_mean_gt_sd_seed_level"] for g in omitted},
            "omitted_have_all_five_same_sign": {gkey(g): robust_flags[gkey(g)]["all_five_same_sign"] for g in omitted},
        },
        "selection_flow": selection,
        "expected_overall_missingness_preservation": preservation,
    }
    out = OUT_DIR / "r6_simulation_uncertainty.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print("WROTE", out)
    print("\n=== full nine-multiplier table (mean +/- seed SD, five seeds) ===")
    print("gap      " + "".join(f"{c:>16s}" for c in PRIMARY))
    for r in table7_full:
        line = f"{r['target_gap_multiplier']:>5g}  "
        for c in PRIMARY:
            line += f"{r[c]['mean']:.4f}±{r[c]['sd']:.4f} "
        print(line)
    print("\n=== robustness of C025-B025 by multiplier ===")
    for g in gaps:
        r = paired["C025_minus_B025"][gkey(g)]
        print(f"gap {g:>5g}: mean={r['mean']:+.5f} sd={r['sd_seed_level_ddof1']:.5f} "
              f"t95=[{r['t95_interval_mean_diff'][0]:+.5f},{r['t95_interval_mean_diff'][1]:+.5f}] "
              f"sign={r['n_positive_values']}P/{r['n_negative_values']}N "
              f"|mean|>SD={r['abs_mean_gt_sd_seed_level']}")
    print("\n=== expected overall missingness rate preserved ===")
    for p in preservation:
        print(p["data_seed"], p["expected_overall_missing_range"], p["actual_missing_at_gap_1"], p["actual_missing_at_gap_-1"])


if __name__ == "__main__":
    main()
