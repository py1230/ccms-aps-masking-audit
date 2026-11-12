"""A3 模型逐项核验：加载已保存的 30 个模型，复算概率与 AP，与参考产物比对。

不训练任何模型、不修改参考预测、不改历史 AP。若不一致，如实保留差异并给出可能原因。

比对对象：
  1. `preds_s*.npz` 里的六组概率（C/B × 2026/2027/2028）；
  2. `a3_persample_predictions.csv` 的 270,000 行概率；
  3. `cells_s*.json` 的 AP；
  4. `results_seed/*.json` 的 30 条历史 AP（只比对，不参与拟合）。

同时检查：样本映射（source_row_0based == val_rows[val_index]、y_true == y_val）、
模型 sha256、M3 评价矩阵 sha256 是否与重建记录一致。

用法：
    python3 verify_a3_models.py --data <aps csv> --repro <复现件目录> --rebuilt <重建目录> --out <输出目录> [--code <依赖代码目录>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--repro", default=None, help="含 model_*.joblib 与 preds_s*.npz 的目录")
    ap.add_argument("--rebuilt", default=None, help="事后重建目录（eval_masks_s*.npz / imputer_s*.joblib）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--code", default=None, help="依赖代码目录（aps_io.py 等）")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent          # .../03_.../scripts
    pkg = here.parent                               # .../03_...
    data = Path(args.data) if args.data else (pkg / "data" / "aps_failure_training_set.csv")
    repro = Path(args.repro) if args.repro else (pkg.parent / "02_11-12复现产物" / "A3_repro_2026-11-12")
    rebuilt = Path(args.rebuilt) if args.rebuilt else (pkg / "rebuilt")
    out = Path(args.out) if args.out else (pkg / "verify_out")
    code = Path(args.code) if args.code else (pkg / "code")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))

    from aps_io import dev_split, load_aps_csv, split_features
    from ccms import rebuild_missing_flags
    from run_cb_firstlook import build_eval_inputs

    t0 = time.time()
    df, meta = load_aps_csv(str(data))
    X, y = split_features(df)
    print(f"[load] X={X.shape} pos={int(y.sum())} sha={meta['sha256'][:12]}", flush=True)

    csv = pd.read_csv(repro / "a3_persample_predictions.csv")
    print(f"[csv] rows={len(csv)}", flush=True)

    cells = []
    map_issues = []
    for s in (11, 22, 33, 44, 55):
        sp = dev_split(y, seed=s, stratify=True)
        va, yva = sp["val"], y[sp["val"]]
        Xva = X[va]

        em = np.load(rebuilt / f"eval_masks_s{s}.npz")
        assert np.array_equal(va.astype(np.int64), em["val_rows"]), f"split {s}: val_rows 不一致"
        shape = tuple(int(v) for v in em["mask_shape"])
        n_cells = shape[0] * shape[1]
        native = np.unpackbits(em["native_bits"])[:n_cells].reshape(shape).astype(bool)
        added = np.unpackbits(em["added_bits"])[:n_cells].reshape(shape).astype(bool)
        merged = np.unpackbits(em["merged_bits"])[:n_cells].reshape(shape).astype(bool)

        ev, _ = build_eval_inputs(Xva)["S1_30_101"]
        ev_nan = np.isnan(ev)
        mask_ok = bool(
            np.array_equal(ev_nan, merged)
            and np.array_equal(np.isnan(Xva), native)
            and np.array_equal(merged, native | added)
        )

        imp = __import__("joblib").load(rebuilt / f"imputer_s{s}.joblib")
        Mv = np.hstack([imp.transform(ev), rebuild_missing_flags(ev)])
        m3_sha = sha_bytes(np.ascontiguousarray(Mv).tobytes())
        m3_match = m3_sha == str(em["m3_eval_sha256"][0])

        zp = np.load(repro / f"preds_s{s}.npz")
        z_val_rows = zp["val_rows"]
        z_y = zp["y_val"]
        sub = csv[csv["split_seed"] == s]
        # 样本映射检查（用第一格做全量核对）
        one = sub[(sub["arm"] == "C") & (sub["model_seed"] == 2026)].sort_values("val_index")
        idx_ok = bool(np.array_equal(one["val_index"].to_numpy(), np.arange(len(va))))
        src_ok = bool(np.array_equal(one["source_row_0based"].to_numpy(), va.astype(np.int64)))
        y_ok = bool(np.array_equal(one["y_true"].to_numpy(), yva.astype(np.int64)))
        npz_map_ok = bool(np.array_equal(z_val_rows, va) and np.array_equal(z_y, yva))
        if not all([idx_ok, src_ok, y_ok, npz_map_ok, mask_ok]):
            map_issues.append({"split": s, "val_index": idx_ok, "source_row": src_ok, "y_true": y_ok,
                               "npz_val_rows": npz_map_ok, "eval_mask": mask_ok})

        cj = json.loads((repro / f"cells_s{s}.json").read_text(encoding="utf-8"))["cells"]
        rec = {(c["arm"], c["model_seed"]): c for c in cj}

        for arm in ("C", "B"):
            for ms in (2026, 2027, 2028):
                mf = repro / f"model_s{s}_{arm}_{ms}.joblib"
                m = __import__("joblib").load(mf)
                prob = m.predict_proba(Mv)[:, 1]

                p_npz = zp[f"{arm}_{ms}"]
                d_npz = float(np.max(np.abs(prob - p_npz)))
                sl = sub[(sub["arm"] == arm) & (sub["model_seed"] == ms)].sort_values("val_index")
                p_csv = sl["prob"].to_numpy()
                d_csv = float(np.max(np.abs(prob - p_csv)))

                ap_new = float(average_precision_score(yva, prob))
                ap_npz = float(average_precision_score(yva, p_npz))
                ap_csv = float(average_precision_score(yva, p_csv))
                hist = rec[(arm, ms)]["recorded_ap"]
                model_sha = sha_file(mf)
                cells.append(
                    {
                        "split": s,
                        "arm": arm,
                        "model_seed": ms,
                        "n": int(len(yva)),
                        "max_abs_prob_diff_vs_npz": d_npz,
                        "max_abs_prob_diff_vs_csv": d_csv,
                        "ap_recomputed": ap_new,
                        "ap_from_npz": ap_npz,
                        "ap_from_csv": ap_csv,
                        "ap_history_recorded": hist,
                        "ap_diff_vs_history": abs(ap_new - hist),
                        "ap_diff_npz_vs_csv": abs(ap_npz - ap_csv),
                        "bit_equal_vs_history": bool(ap_new == hist),
                        "model_sha256": model_sha,
                        "model_sha256_matches_cells_json": model_sha == rec[(arm, ms)]["model_sha256"],
                        "mask_sha256_matches": rec[(arm, ms)]["mask_sha256"] == str(zp["mask_sha256"][0 if arm == "C" else 1]),
                    }
                )
                print(
                    f"[verify] s{s} {arm} {ms}: dP_npz={d_npz:.3e} dP_csv={d_csv:.3e} "
                    f"AP={ap_new!r} hist={hist!r} equal={ap_new == hist}",
                    flush=True,
                )

    max_p_npz = max(c["max_abs_prob_diff_vs_npz"] for c in cells)
    max_p_csv = max(c["max_abs_prob_diff_vs_csv"] for c in cells)
    max_ap_hist = max(c["ap_diff_vs_history"] for c in cells)
    n_equal = sum(c["bit_equal_vs_history"] for c in cells)

    result = {
        "identity": "核验记录：加载 2026-11-12 复现件已保存的 30 个模型，复算概率与 AP；未重新训练",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "script_sha256": sha_file(Path(__file__).resolve()),
        },
        "data": {"path": str(data), "sha256": meta["sha256"], "n_rows": int(len(df))},
        "inputs": {
            "repro_dir": str(repro),
            "rebuilt_dir": str(rebuilt),
            "preds_sha256": {f"preds_s{s}.npz": sha_file(repro / f"preds_s{s}.npz") for s in (11, 22, 33, 44, 55)},
            "csv_sha256": sha_file(repro / "a3_persample_predictions.csv"),
        },
        "summary": {
            "n_cells": len(cells),
            "max_abs_prob_diff_vs_npz": max_p_npz,
            "max_abs_prob_diff_vs_csv": max_p_csv,
            "max_ap_diff_vs_history": max_ap_hist,
            "n_ap_bit_equal_vs_history": n_equal,
            "sample_mapping_ok": len(map_issues) == 0,
            "mapping_issues": map_issues,
            "all_model_sha_match": all(c["model_sha256_matches_cells_json"] for c in cells),
            "all_mask_sha_match": all(c["mask_sha256_matches"] for c in cells),
            "wall_seconds": round(time.time() - t0, 2),
        },
        "cells": cells,
    }
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[summary] max|ΔP| vs npz = {max_p_npz:.3e}; vs csv = {max_p_csv:.3e}; "
          f"max|ΔAP| vs history = {max_ap_hist:.3e}; equal {n_equal}/30; mapping_ok={len(map_issues)==0}",
          flush=True)
    print(f"[done] {out/'verification.json'} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
