#!/usr/bin/env python3
"""Recompute the point estimates of the main tables from the archived per-sample
probabilities in this package, without retraining.

Run from anywhere:  python3 10_repro_check_2026-11-05/reproduce_tables.py

Paths are resolved from the package root, so the package can be unpacked under
any directory name. The archived script revision/analysis/
lead_r11_manifest_and_reproduce.py of 2026-11-01 assumed the original folder
name; this file replaces it for that purpose only.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "08_repro_bundle", "02_本轮代码与结果", "results")
FAMILIES = {
    "S0": ["S0"],
    "S1_30": ["S1_30_101", "S1_30_202", "S1_30_303"],
    "S2": ["S2_1101", "S2_1202", "S2_1303"],
    "S3": ["S3_2101", "S3_2202", "S3_2303"],
}
ARMS = {"M3": "M3", "C(0.25)": "C025", "B(0.25)": "B025"}
PAPER = {
    ("val", "M3"): [0.8617, 0.6239, 0.6310, 0.6841],
    ("val", "C(0.25)"): [0.8650, 0.5320, 0.6047, 0.6611],
    ("val", "B(0.25)"): [0.8744, 0.6620, 0.6525, 0.7139],
    ("test", "M3"): [0.8910, 0.7099, 0.6833, 0.7513],
    ("test", "C(0.25)"): [0.8932, 0.6234, 0.6774, 0.7284],
    ("test", "B(0.25)"): [0.9009, 0.7483, 0.7076, 0.7838],
}


def ap(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main():
    files = sorted(glob.glob(os.path.join(RES, "aps_seed*.npz")))
    if len(files) != 5:
        sys.exit(f"expected five aps_seed*.npz, found {len(files)} in {RES}")
    cells, diffs = [], []
    for split in ("val", "test"):
        for arm_paper, arm_npz in ARMS.items():
            means = []
            for fam, conds in FAMILIES.items():
                vals = []
                for f in files:
                    d = np.load(f, allow_pickle=True)
                    y = d["y_val"] if split == "val" else d["y_test"]
                    for c in conds:
                        vals.append(ap(y, d[f"{arm_npz}__{split}__{c}"]))
                means.append(float(np.mean(vals)))
            for fam, got, want in zip(FAMILIES, means, PAPER[(split, arm_paper)]):
                cells.append({"split": split, "arm": arm_paper, "family": fam,
                              "recomputed": round(got, 6), "manuscript": want,
                              "abs_diff": abs(got - want)})
                diffs.append(abs(got - want))
    worst = max(diffs)
    res = {
        "script": os.path.abspath(__file__),
        "package_root": ROOT,
        "source": "08_repro_bundle/02_本轮代码与结果/results/aps_seed{11,22,33,44,55}.npz",
        "target": "Table 4 (mean AP on APS at fixed alpha = 0.25), 24 cells",
        "n_cells": len(cells),
        "max_abs_diff": worst,
        "agrees_at_printed_precision": bool(worst < 1e-4),
        "cells": cells,
    }
    out = os.path.join(HERE, "table4_recomputed.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("n_cells", "max_abs_diff", "agrees_at_printed_precision")}, indent=1))
    print("written:", out)
    return 0 if worst < 1e-4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
