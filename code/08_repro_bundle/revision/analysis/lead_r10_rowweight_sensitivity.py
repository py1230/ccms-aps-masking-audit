"""R10: row-weight definition + sensitivity for the APS validation conditional intervals.

Variant A (current, published): unique union of the five validation splits' original
row ids is the resampling unit; each split's AP is computed with the union weights
mapped onto its own rows, so rows shared by several splits carry one shared weight.

Variant B (sensitivity): each split is resampled independently within split
(stratified by label); the estimand remains the mean of the five per-split AP
differences.

Both variants use 1000 draws and seed 20261101; variant A must reproduce the saved
results/aps_matched_bootstrap.json intervals exactly.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parents[2]  # bundle root
RES = ROOT / "CCMS论文修订完整资料包_2026-11-01_192149/02_本轮代码与结果/results"
OUT = ROOT / "revision/analysis"
SEEDS = [11, 22, 33, 44, 55]
FAM = {
    "S0": ["S0"],
    "S1_30": ["S1_30_101", "S1_30_202", "S1_30_303"],
    "S2": ["S2_1101", "S2_1202", "S2_1303"],
    "S3": ["S3_2101", "S3_2202", "S3_2303"],
}
NBOOT = 1000
BOOT_SEED = 20261101


def prep(y, p):
    o = np.argsort(-p, kind="stable")
    last = np.r_[np.flatnonzero(np.diff(p[o]) != 0), len(y) - 1]
    return o, y[o], last


def wap(pr, w):
    o, y, last = pr
    ww = w[o]
    tp = np.cumsum(ww * y)[last]
    den = np.cumsum(ww)[last]
    total = tp[-1]
    if total <= 0:
        raise ValueError("zero bootstrap positives")
    return float(np.sum(np.diff(np.r_[0.0, tp]) * np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)) / total)


def interval(v):
    return [float(z) for z in np.quantile(v, [0.025, 0.975])]


def main():
    packs = [np.load(RES / f"aps_seed{s}.npz") for s in SEEDS]
    saved = json.loads((RES / "aps_matched_bootstrap.json").read_text())
    out = {
        "generated_at": "2026-11-01",
        "n_bootstrap": NBOOT,
        "seed": BOOT_SEED,
        "variant_A": "shared union weights over unique original validation row ids (current)",
        "variant_B": "independent within-split label-stratified resampling of each validation split",
        "estimand": "mean over the five splits of the per-split mean AP difference C(0.25)-B(0.25), stress copies averaged first",
        "reproduction_check": {},
        "results": {},
    }
    for dataset in ["val", "test"]:
        ys = [z["y_" + dataset] for z in packs]
        ids = [z["val_rows"] for z in packs] if dataset == "val" else [np.arange(len(ys[0])) for _ in packs]
        union = np.unique(np.concatenate(ids))
        uy = np.empty(len(union), int)
        mappings = [np.searchsorted(union, v) for v in ids]
        for y, m in zip(ys, mappings):
            uy[m] = y
        prs = {}
        points = {}
        for f, conds in FAM.items():
            prs[f] = [
                [(prep(y, z["C025__" + dataset + "__" + k]), prep(y, z["B025__" + dataset + "__" + k])) for k in conds]
                for z, y in zip(packs, ys)
            ]
            points[f] = [
                float(np.mean([AP(y, z["C025__" + dataset + "__" + k]) - AP(y, z["B025__" + dataset + "__" + k]) for k in conds]))
                for z, y in zip(packs, ys)
            ]

        drawsA = {f: np.zeros((NBOOT, 5)) for f in FAM}
        rng = np.random.default_rng(BOOT_SEED)
        pidx = np.flatnonzero(uy == 1)
        nidx = np.flatnonzero(uy == 0)
        for b in range(NBOOT):
            ix = np.r_[rng.choice(pidx, len(pidx), replace=True), rng.choice(nidx, len(nidx), replace=True)]
            weights = np.bincount(ix, minlength=len(union)).astype(float)
            for f in FAM:
                for si, m in enumerate(mappings):
                    w = weights[m]
                    drawsA[f][b, si] = np.mean([wap(a, w) - wap(c, w) for a, c in prs[f][si]])

        drawsB = {f: np.zeros((NBOOT, 5)) for f in FAM}
        rngB = np.random.default_rng(BOOT_SEED)
        split_pos = [np.flatnonzero(y == 1) for y in ys]
        split_neg = [np.flatnonzero(y == 0) for y in ys]
        for b in range(NBOOT):
            for si in range(5):
                ip = rngB.choice(split_pos[si], len(split_pos[si]), replace=True)
                ineg = rngB.choice(split_neg[si], len(split_neg[si]), replace=True)
                w = np.bincount(np.r_[ip, ineg], minlength=len(ys[si])).astype(float)
                for f in FAM:
                    drawsB[f][b, si] = np.mean([wap(a, w) - wap(c, w) for a, c in prs[f][si]])

        out["results"][dataset] = {}
        for f in FAM:
            rep = saved["results"][dataset][f]
            out["results"][dataset][f] = {
                "point_mean": float(np.mean(points[f])),
                "A_shared_union": {
                    "ci95": interval(drawsA[f].mean(1)),
                    "per_split_delta": points[f],
                    "per_split_ci": [interval(drawsA[f][:, i]) for i in range(5)],
                },
                "B_independent_split": {
                    "ci95": interval(drawsB[f].mean(1)),
                    "per_split_ci": [interval(drawsB[f][:, i]) for i in range(5)],
                },
            }
            if dataset in saved["results"] and f in saved["results"][dataset]:
                a = out["results"][dataset][f]["A_shared_union"]["ci95"]
                out["reproduction_check"][f"{dataset}_{f}"] = {
                    "point_abs_diff": abs(out["results"][dataset][f]["point_mean"] - rep["delta_mean"]),
                    "ci_abs_max_diff": max(abs(a[0] - rep["ci95"][0]), abs(a[1] - rep["ci95"][1])),
                }

    (OUT / "lead_r10_rowweight_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out["reproduction_check"], indent=2))
    for ds in out["results"]:
        for f, v in out["results"][ds].items():
            print(ds, f, "point", round(v["point_mean"], 4),
                  "A", [round(x, 4) for x in v["A_shared_union"]["ci95"]],
                  "B", [round(x, 4) for x in v["B_independent_split"]["ci95"]])


if __name__ == "__main__":
    main()
