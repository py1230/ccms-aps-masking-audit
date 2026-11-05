"""Summarise the additional experiments: aggregate by experiment/arm/split and write markdown tables plus JSON."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("CCMS_OUT", os.path.join(_HERE, "results"))
OUT = os.environ.get("CCMS_SUMMARY", os.path.join(_HERE, "summary.md"))
HEAD = ("S0", "S1_30_101", "S2_1101", "S3_2101")
SPLITS = (11, 22, 33, 44, 55)


def load():
    recs = []
    for p in glob.glob(os.path.join(RES, "*.json")):
        recs.append(json.load(open(p)))
    if not recs:
        raise SystemExit(
            "No JSON records found in %s. Run run_new_experiments.py first, or point "
            "CCMS_OUT at the directory that holds its output." % RES
        )
    return recs


def agg(recs, exp):
    sub = [r for r in recs if r["exp"] == exp]
    if not sub:
        return None
    out = {}
    for arm in ("none", "C", "B"):
        per_split = {}
        for s in SPLITS:
            rows = [r for r in sub if r["split"] == s and r["arm"] == arm]
            if not rows:
                continue
            per_split[s] = {c: float(np.mean([r["ap"][c] for r in rows])) for c in HEAD}
        if per_split:
            out[arm] = per_split
    # contrasts per split
    for a, b in (("C", "B"), ("C", "none"), ("B", "none")):
        if a in out and b in out:
            d = {}
            for s in SPLITS:
                if s in out[a] and s in out[b]:
                    d[s] = {c: out[a][s][c] - out[b][s][c] for c in HEAD}
            out[f"{a}-{b}"] = d
    return out


def fmt(x):
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def main():
    recs = load()
    lines = ["# Summary of the additional experiments", ""]
    summary = {}
    for exp in ("A", "B", "C", "D"):
        ag = agg(recs, exp)
        if not ag:
            continue
        summary[exp] = ag
        backbone = {"A": "M4 (native NaN)", "B": "M3 (early stopping off)",
                    "C": "missingness-only", "D": "ExtraTrees on M3 inputs"}[exp]
        lines += [f"## Experiment {exp}: {backbone}", ""]
        for cond in HEAD:
            lines += [f"### {cond}", "",
                      "| split | none | C | B | C−B | C−none |",
                      "|---|---|---|---|---|---|"]
            for s in SPLITS:
                row = [str(s)]
                for arm in ("none", "C", "B"):
                    row.append(f"{ag[arm][s][cond]:.4f}" if arm in ag and s in ag[arm] else "—")
                row.append(fmt(ag["C-B"][s][cond]) if "C-B" in ag and s in ag["C-B"] else "—")
                row.append(fmt(ag["C-none"][s][cond]) if "C-none" in ag and s in ag["C-none"] else "—")
                lines.append("| " + " | ".join(row) + " |")
            mean = {k: float(np.mean([v[cond] for v in ag[k].values()])) if k in ag else None
                    for k in ("none", "C", "B", "C-B", "C-none")}
            lines.append(f"| **mean** | {mean['none']:.4f} | {mean['C']:.4f} | {mean['B']:.4f} | "
                         f"**{mean['C-B']:+.4f}** | **{mean['C-none']:+.4f}** |")
            nneg = sum(1 for s in SPLITS if "C-B" in ag and s in ag["C-B"] and ag["C-B"][s][cond] < 0)
            lines += ["", f"Splits with negative C-B: {nneg}/5", ""]
        lines.append("")
    open(OUT, "w").write("\n".join(lines))
    json.dump(summary, open(os.path.join(os.path.dirname(OUT), "summary.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
