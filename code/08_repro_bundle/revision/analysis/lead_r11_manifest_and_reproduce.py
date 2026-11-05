"""R11: build the artifact manifest (SHA-256) and demonstrate one independently
reproducible audit conclusion from the packaged artifacts alone.

The check recomputes the Table-3 fixed-alpha mean AP values and the Table-4
C-B point differences directly from the saved per-row predictions, without
retraining, and compares them with the saved summary JSON.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP

ROOT = Path(__file__).resolve().parents[2]  # bundle root
PKG = ROOT / "CCMS论文修订完整资料包_2026-11-01_192149"
RUN = PKG / "02_本轮代码与结果"
RES = RUN / "results"
OUT = ROOT / "revision/analysis"
SEEDS = [11, 22, 33, 44, 55]
FAM = {
    "S0": ["S0"],
    "S1_30": ["S1_30_101", "S1_30_202", "S1_30_303"],
    "S2": ["S2_1101", "S2_1202", "S2_1303"],
    "S3": ["S3_2101", "S3_2202", "S3_2303"],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    # --- 1. manifest of the supplementary package -------------------------
    files = []
    for sub in [
        "02_本轮代码与结果/results",
        "02_本轮代码与结果/scripts",
        "02_本轮代码与结果/figures",
        "02_本轮代码与结果/protocol",
        "03_原始项目归档/CCMS核查_2026-10-31/records",
        "03_原始项目归档/CCMS核查_2026-10-31/src",
        "03_原始项目归档/CCMS核查_2026-10-31/protocol",
    ]:
        base = PKG / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.suffix != ".nc":
                files.append(
                    {
                        "path": str(p.relative_to(PKG)),
                        "bytes": p.stat().st_size,
                        "sha256": sha(p),
                    }
                )
    # revision-side artifacts (this round's new analyses, figures, manuscript)
    revision_files = []
    for sub in ["revision/analysis", "revision/figures", "revision/build", "revision/notes"]:
        base = ROOT / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts:
                revision_files.append({
                    "path": str(p.relative_to(ROOT)),
                    "bytes": p.stat().st_size,
                    "sha256": sha(p),
                })
    EXCLUDE = {"r11_manifest.json", "r11_reproduce_check.json"}
    revision_files = [f for f in revision_files if Path(f["path"]).name not in EXCLUDE]
    manifest = {
        "generated_at": "2026-11-01",
        "package_version": "CCMS-revision-2026-11-01",
        "persistent_id": None,
        "persistent_id_note": "the supplementary package accompanies the submission; the repository and DOI identifiers are added by the authors in the submission system",
        "n_files_package": len(files),
        "n_files_revision": len(revision_files),
        "n_files": len(files) + len(revision_files),
        "package_files": files,
        "revision_files": revision_files,
    }
    (OUT / "r11_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # --- 2. reproduce one audit conclusion from artifacts only ------------
    packs = [np.load(RES / f"aps_seed{s}.npz") for s in SEEDS]
    reproduced = {}
    for dataset in ["val", "test"]:
        ys = [z["y_" + dataset] for z in packs]
        reproduced[dataset] = {}
        for cfg in ["M3", "C025", "B025"]:
            vals = {}
            for f, conds in FAM.items():
                per_split = []
                for z, y in zip(packs, ys):
                    per_split.append(float(np.mean([AP(y, z[f"{cfg}__{dataset}__{k}"]) for k in conds])))
                vals[f] = {"mean_five_splits": float(np.mean(per_split)), "per_split": per_split}
            reproduced[dataset][cfg] = vals

    # cross-check against aps_reconstruction.json (saved metrics) and table 4
    recon = json.loads((RES / "aps_reconstruction.json").read_text())
    check = {}
    for dataset in ["val", "test"]:
        for cfg in ["M3", "C025", "B025"]:
            for f, conds in FAM.items():
                saved = float(
                    np.mean(
                        [
                            np.mean([s["metrics"][cfg][dataset][k] for k in conds])
                            for s in recon["splits"]
                        ]
                    )
                )
                check[f"{cfg}__{dataset}__{f}"] = {
                    "recomputed": reproduced[dataset][cfg][f]["mean_five_splits"],
                    "from_aps_reconstruction_json": saved,
                    "abs_diff": abs(reproduced[dataset][cfg][f]["mean_five_splits"] - saved),
                }
    boot = json.loads((RES / "aps_matched_bootstrap.json").read_text())
    cb = {}
    for dataset in ["val", "test"]:
        for f in FAM:
            cb[f"{dataset}__{f}"] = {
                "recomputed_C_minus_B": reproduced[dataset]["C025"][f]["mean_five_splits"]
                - reproduced[dataset]["B025"][f]["mean_five_splits"],
                "saved_delta_mean": boot["results"][dataset][f]["delta_mean"],
            }
            cb[f"{dataset}__{f}"]["abs_diff"] = abs(
                cb[f"{dataset}__{f}"]["recomputed_C_minus_B"] - cb[f"{dataset}__{f}"]["saved_delta_mean"]
            )
    result = {
        "generated_at": "2026-11-01",
        "python": platform.python_version(),
        "checks": {
            "table3_mean_ap_vs_aps_reconstruction_json": check,
            "table4_point_difference_vs_aps_matched_bootstrap_json": cb,
        },
        "max_abs_diff_table3": max(v["abs_diff"] for v in check.values()),
        "max_abs_diff_table4": max(v["abs_diff"] for v in cb.values()),
    }
    (OUT / "r11_reproduce_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("files in manifest:", len(files), "+ revision", len(revision_files))
    print("max |table3 diff| =", result["max_abs_diff_table3"])
    print("max |table4 diff| =", result["max_abs_diff_table4"])
    for k, v in list(cb.items())[:8]:
        print(k, "recomputed", round(v["recomputed_C_minus_B"], 6), "saved", round(v["saved_delta_mean"], 6))


if __name__ == "__main__":
    main()
