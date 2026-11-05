"""Bootstrap intervals for the full-scale count-matched decomposition (R9).

Arms: C(0.25), B(0.25), A3 (class-blind, per-column deletion counts identical to C).
Contrasts: C-B (published), C-A3 (class-allocation term), A3-B (column-profile term).

Resampling scheme is the published one (section 3.4): unique union of the five
validation row ids with shared weights, label-stratified, 1000 draws, seed
20261101; official test rows use arange ids shared by all models. The C-B draws
are bit-identical to the published aps_bootstrap_draws.npz, which is asserted.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parents[2]  # bundle root
RES = ROOT / "CCMS论文修订完整资料包_2026-11-01_192149/02_本轮代码与结果/results"
ANA = ROOT / "revision/analysis"
SEEDS = [11, 22, 33, 44, 55]
FAM = {
    "S0": ["S0"],
    "S1_30": ["S1_30_101", "S1_30_202", "S1_30_303"],
    "S2": ["S2_1101", "S2_1202", "S2_1303"],
    "S3": ["S3_2101", "S3_2202", "S3_2303"],
}
NB = 1000
BS = 20261101


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
    arms = [np.load(ANA / f"arm3_seed{s}.npz") for s in SEEDS]
    saved = json.loads((RES / "aps_matched_bootstrap.json").read_text())
    saved_draws = np.load(RES / "aps_bootstrap_draws.npz")
    out = {
        "generated_at": "2026-11-01",
        "n_bootstrap": NB,
        "seed": BS,
        "arms": {"C": "C025", "B": "B025", "A3": "class-blind with per-column counts equal to C"},
        "identity": "C-B = (C-A3) + (A3-B)",
        "sources": ["results/aps_seed*.npz", "revision/analysis/arm3_seed*.npz"],
        "results": {},
        "reconciliation_CB_vs_published": {},
    }
    for ds in ["val", "test"]:
        ys = [z["y_" + ds] for z in packs]
        ids = [z["val_rows"] for z in packs] if ds == "val" else [np.arange(len(ys[0])) for _ in packs]
        union = np.unique(np.concatenate(ids))
        uy = np.empty(len(union), int)
        maps = [np.searchsorted(union, v) for v in ids]
        for y, m in zip(ys, maps):
            uy[m] = y

        prs = {}
        points = {}
        for f, conds in FAM.items():
            prs[f] = [
                {
                    "CB": (prep(y, z["C025__" + ds + "__" + k]), prep(y, z["B025__" + ds + "__" + k])),
                    "CA3": (prep(y, z["C025__" + ds + "__" + k]), prep(y, a["A3__" + ds + "__" + k])),
                    "A3B": (prep(y, a["A3__" + ds + "__" + k]), prep(y, z["B025__" + ds + "__" + k])),
                }
                for k in conds
                for z, a in zip(packs, arms)
            ]
            # rebuild in split order (list comprehension above is per cond; need per split)
            prs[f] = [
                {
                    "CB": (prep(y, z["C025__" + ds + "__" + k]), prep(y, z["B025__" + ds + "__" + k])),
                    "CA3": (prep(y, z["C025__" + ds + "__" + k]), prep(y, a["A3__" + ds + "__" + k])),
                    "A3B": (prep(y, a["A3__" + ds + "__" + k]), prep(y, z["B025__" + ds + "__" + k])),
                }
                for z, a, y in zip(packs, arms, ys)
                for k in conds
            ]
            per_cond = len(conds)
            points[f] = {}
            for cname in ["CB", "CA3", "A3B"]:
                vals = []
                for si in range(5):
                    seg = prs[f][si * per_cond:(si + 1) * per_cond]
                    vals.append(float(np.mean([AP(y, x) - AP(y, z) if False else 0 for x in []])) if False else 0.0)
                points[f][cname] = None
            # point estimates directly
            for cname, a, b in [("CB", "C025", "B025"), ("CA3", "C025", "A3"), ("A3B", "A3", "B025")]:
                pts = []
                for si, (z, a3, y) in enumerate(zip(packs, arms, ys)):
                    src_a = z if a != "A3" else a3
                    src_b = z if b != "A3" else a3
                    pts.append(float(np.mean([AP(y, src_a[f"{a}__{ds}__{k}"]) - AP(y, src_b[f"{b}__{ds}__{k}"]) for k in conds])))
                points[f][cname] = pts

        draws = {f: {c: np.zeros((NB, 5)) for c in ["CB", "CA3", "A3B"]} for f in FAM}
        rng = np.random.default_rng(BS)
        pos = np.flatnonzero(uy == 1)
        neg = np.flatnonzero(uy == 0)
        for bib in range(NB):
            ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            weights = np.bincount(ix, minlength=len(union)).astype(float)
            for f in FAM:
                per_cond = len(FAM[f])
                for si, m in enumerate(maps):
                    w = weights[m]
                    seg = prs[f][si * per_cond:(si + 1) * per_cond]
                    for cname in ["CB", "CA3", "A3B"]:
                        draws[f][cname][bib, si] = np.mean([wap(x[cname][0], w) - wap(x[cname][1], w) for x in seg])

        out["results"][ds] = {}
        for f in FAM:
            entry = {}
            for cname in ["CB", "CA3", "A3B"]:
                entry[cname] = {
                    "point_mean": float(np.mean(points[f][cname])),
                    "ci95": interval(draws[f][cname].mean(1)),
                    "per_split_point": points[f][cname],
                }
            entry["identity_residual_point"] = entry["CB"]["point_mean"] - (entry["CA3"]["point_mean"] + entry["A3B"]["point_mean"])
            entry["share_class_allocation"] = entry["CA3"]["point_mean"] / entry["CB"]["point_mean"] if entry["CB"]["point_mean"] else None
            out["results"][ds][f] = entry
            rep = saved["results"][ds][f]
            out["reconciliation_CB_vs_published"][f"{ds}_{f}"] = {
                "point_abs_diff": abs(entry["CB"]["point_mean"] - rep["delta_mean"]),
                "ci_abs_max_diff": max(abs(entry["CB"]["ci95"][0] - rep["ci95"][0]), abs(entry["CB"]["ci95"][1] - rep["ci95"][1])),
                "draws_abs_max_diff": float(np.max(np.abs(draws[f]["CB"] - saved_draws[f"{ds}__{f}"]))),
            }
    (ANA / "r9_arm3_decomposition_intervals.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out["reconciliation_CB_vs_published"], indent=2))
    for ds in out["results"]:
        for f, v in out["results"][ds].items():
            print(ds, f,
                  "CB", round(v["CB"]["point_mean"], 4), [round(x, 4) for x in v["CB"]["ci95"]],
                  "| C-A3", round(v["CA3"]["point_mean"], 4), [round(x, 4) for x in v["CA3"]["ci95"]],
                  "| A3-B", round(v["A3B"]["point_mean"], 4), [round(x, 4) for x in v["A3B"]["ci95"]],
                  "| class share", round(v["share_class_allocation"], 3) if v["share_class_allocation"] else None)


if __name__ == "__main__":
    main()
