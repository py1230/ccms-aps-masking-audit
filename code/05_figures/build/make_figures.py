"""Regenerate the four manuscript figures with unified configuration notation.

Inputs (all saved results, nothing is recomputed from raw predictions here):
  revision/analysis/r13_threeway.json                    -> fig 1 (three-way)
  package secom_summary.json                             -> fig 2 (SECOM folds)
  revision/analysis/r9_deletion_diagnostics.json         -> fig 3 (a)(b)
  revision/analysis/r9_arm3_decomposition_intervals.json -> fig 3 (c)
  revision/analysis/r6_simulation_uncertainty.json       -> fig 4 (9 multipliers)

Notation: M3, M4, C(0.25), B(0.25), C(1), B(1), C(sel), B(sel) -- identical to
the tables in the manuscript.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # bundle root
ANA = ROOT / "revision/analysis"
PKG = ROOT / "CCMS论文修订完整资料包_2026-11-01_192149/02_本轮代码与结果/results"
OUT = ROOT / "revision/figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})
COL = {
    "M3": "#333333",
    "M4": "#0072B2",
    "C(0.25)": "#D55E00",
    "B(0.25)": "#009E73",
    "C(1)": "#CC79A7",
    "B(1)": "#56B4E9",
    "C(sel)": "#D55E00",
    "B(sel)": "#009E73",
    "C-B": "#0072B2",
    "C-M3": "#D55E00",
    "B-M3": "#009E73",
}


def read(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- fig 1
def fig1():
    d = read(ANA / "r13_threeway.json")["results"]
    rows = [("val", "S0", "Val S0"), ("val", "S1_30", "Val S1 30%"), ("val", "S2", "Val S2"), ("val", "S3", "Val S3"),
            ("test", "S0", "Test S0"), ("test", "S1_30", "Test S1 30%"), ("test", "S2", "Test S2"), ("test", "S3", "Test S3")]
    contrasts = [("C-B", "C(0.25) - B(0.25)"), ("C-M3", "C(0.25) - M3"), ("B-M3", "B(0.25) - M3")]
    fig, ax = plt.subplots(figsize=(6.9, 3.6), layout="constrained")
    offsets = [-0.22, 0.0, 0.22]
    for k, (key, label) in enumerate(contrasts):
        for i, (ds, fam, _) in enumerate(rows):
            v = d[ds][fam][key]
            delta = v["delta_mean"]
            lo, hi = v["ci95"]
            ax.errorbar(delta, i + offsets[k], xerr=[[delta - lo], [hi - delta]],
                        fmt="o", ms=3.4, color=COL[key], ecolor=COL[key], elinewidth=0.9,
                        capsize=2.2, label=label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean AP difference (five splits, equal weight)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.grid(axis="x", alpha=0.2)
    save(fig, "01_three_way_decomposition")


# ---------------------------------------------------------------- fig 2
def fig2():
    x = read(PKG / "secom_summary.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1), layout="constrained")
    for ax, k, title in [(axs[0], "S0", "(a) Original"), (axs[1], "S1_30", "(b) Extra 30% missing")]:
        for c in ["M3", "M4", "C025", "B025"]:
            v = x["per_config"][c][k]["fold_AP"]
            label = {"M3": "M3", "M4": "M4", "C025": "C(0.25)", "B025": "B(0.25)"}[c]
            ax.plot(range(1, 6), v, marker="o", ms=3, lw=1, color=COL[label], label=label)
        ax.set(title=title, xlabel="Outer fold", ylabel="AP", xticks=range(1, 6), ylim=(0, 0.27))
        ax.grid(alpha=0.2)
    axs[1].legend(frameon=False, fontsize=7)
    save(fig, "02_SECOM_folds")


# ---------------------------------------------------------------- fig 3
def fig3():
    dele = read(ANA / "r9_deletion_diagnostics.json")
    dec = read(ANA / "r9_arm3_decomposition_intervals.json")["results"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained")

    # (a) top-8 share of positive-class deletions, per split and mask seed
    for i, sp in enumerate(dele["splits"]):
        shares = [p["C"]["top8_pos_share_of_pos_deletions"] for p in sp["pairs"]]
        axs[0].scatter([i + 1] * len(shares), shares, s=14, color=COL["C(0.25)"], alpha=0.85,
                       edgecolor="white", linewidth=0.3)
    axs[0].set(title="(a) Top-8 share", xlabel="Split", xticks=range(1, 6),
               ylim=(0.80, 0.95))
    axs[0].set_ylabel("Share of positive deletions")
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axs[0].grid(alpha=0.2)

    # (b) class composition of actual deletions
    pool = dele["pooled_five_splits"]
    pos_share = [100 * pool["C_pos_share"]["five_split_mean"], 100 * pool["B_pos_share"]["five_split_mean"]]
    bars = axs[1].bar([0, 1], pos_share, width=0.55, color=[COL["C(0.25)"], COL["B(0.25)"]])
    for b, v in zip(bars, pos_share):
        axs[1].text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}%", ha="center", fontsize=7.5)
    axs[1].set_xticks([0, 1], ["C(0.25)", "B(0.25)"])
    axs[1].set(title="(b) Positive-class share",
               ylabel="Share of that arm's deletions (%)", ylim=(0, 1.35))
    axs[1].grid(axis="y", alpha=0.2)

    # (c) decomposition
    rows = [("val", "S0", "V S0"), ("val", "S1_30", "V S1 30%"), ("val", "S2", "V S2"), ("val", "S3", "V S3"),
            ("test", "S0", "T S0"), ("test", "S1_30", "T S1 30%"), ("test", "S2", "T S2"), ("test", "S3", "T S3")]
    xs = np.arange(len(rows))
    ca3 = [dec[ds][f]["CA3"]["point_mean"] for ds, f, _ in rows]
    a3b = [dec[ds][f]["A3B"]["point_mean"] for ds, f, _ in rows]
    cb = [dec[ds][f]["CB"]["point_mean"] for ds, f, _ in rows]
    axs[2].bar(xs - 0.19, ca3, width=0.38, color="#D55E00", label="C - A3 (class allocation)")
    axs[2].bar(xs + 0.19, a3b, width=0.38, color="#009E73", label="A3 - B (column profile)")
    axs[2].plot(xs, cb, "k_", ms=11, mew=1.6, label="C - B")
    axs[2].axhline(0, color="black", lw=0.8)
    axs[2].set_xticks(xs, [r[2] for r in rows], rotation=45, ha="right", fontsize=7)
    axs[2].set(title="(c) Count-matched decomposition", ylabel="Mean AP difference", xlabel="Condition")
    axs[2].legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.2)
    axs[2].grid(axis="y", alpha=0.2)
    save(fig, "03_deletion_and_decomposition")


# ---------------------------------------------------------------- fig 4
def fig4():
    d = read(ANA / "r6_simulation_uncertainty.json")
    pc = d["per_config_per_gap"]
    ms = [1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1]

    def series(cfg):
        means = np.array([pc[cfg][f"gap_{m}"]["mean"] for m in ms], dtype=float)
        sds = np.array([pc[cfg][f"gap_{m}"]["sd_seed_level_ddof1"] for m in ms], dtype=float)
        return means, sds

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), layout="constrained")
    for ax, cfgs, title in [(axs[0], ["M3", "C025", "B025", "C100", "B100"], "(a) Fixed alpha"),
                            (axs[1], ["M3", "C_selected", "B_selected"], "(b) Source-selected alpha")]:
        for i, c in enumerate(cfgs):
            mean, sd = series(c)
            label = {"M3": "M3", "C025": "C(0.25)", "B025": "B(0.25)", "C100": "C(1)",
                     "B100": "B(1)", "C_selected": "C(sel)", "B_selected": "B(sel)"}[c]
            ax.plot(ms, mean, color=COL[label], marker=["o", "s", "^", "v", "D", "x", "P"][i % 7],
                    ms=3, lw=1.1, label=label)
            ax.fill_between(ms, mean - sd, mean + sd, color=COL[label], alpha=0.12, lw=0)
        ax.set(title=title, xlabel="Target / source missingness-gap multiplier m", ylabel="AP",
               ylim=(0, 0.62))
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.6, frameon=False)
    save(fig, "04_continuous_multiplier")
    (OUT / "provenance.json").write_text(json.dumps({
        "tool": "matplotlib", "version": matplotlib.__version__,
        "sources": ["r13_threeway.json", "secom_summary.json (package results)",
                    "r9_deletion_diagnostics.json", "r9_arm3_decomposition_intervals.json",
                    "r6_simulation_uncertainty.json"],
        "note": "Fig 1 error bars are nominal 95% conditional bootstrap intervals. Fig 4 error bands are "
                "seed-level standard deviations (ddof=1) over five data seeds, not sampling intervals.",
        "figures": {"01_three_way_decomposition": "fig 1", "02_SECOM_folds": "fig 2",
                    "03_deletion_and_decomposition": "fig 3", "04_continuous_multiplier": "fig 4"},
    }, indent=2))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
