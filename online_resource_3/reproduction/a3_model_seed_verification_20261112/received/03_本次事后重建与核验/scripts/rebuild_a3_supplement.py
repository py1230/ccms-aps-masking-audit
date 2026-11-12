"""事后重建 A3 的评价侧材料（不训练任何模型，不改动原预测）。

用途：2026-11-12 的 A3 复现件保存了 30 个模型与逐样本概率，但**没有保存**
五个验证划分的评价掩码与插补预处理对象。本脚本用同一份原始数据、同一划分索引、
同一组源代码，把这部分补齐，并明确标注为「事后重建的复现补充材料」。

重建内容（每个划分 s ∈ {11,22,33,44,55}）：
1. 评价掩码三项：原生缺失 / 追加删除（S1_30_101，rate=0.30, seed=101）/ 合并后；
2. 只由该划分训练部分拟合的中位数（数组 + 可加载的 MedianImputer 对象）；
3. val_rows（样本行号）、特征顺序、M3 表示的构造记录与副本矩阵哈希；
4. 依赖代码清单与哈希（aps_io / ccms / pipeline / audit_metrics / run_cb_firstlook）。

用法：
    python3 rebuild_a3_supplement.py --project-root <项目根> --out <输出目录>
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
import sklearn


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/user/workspace/默认项目")
    ap.add_argument("--data", default=None, help="APS CSV 路径；缺省用 <project-root>/aps_failure_training_set.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--code", default=None, help="依赖代码目录（aps_io.py 等平铺）；缺省用 <project-root>/CCMS核查_2026-10-31/src 与 /scripts")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    data = Path(args.data) if args.data else root / "aps_failure_training_set.csv"
    out = Path(args.out) if args.out else root / "A3_核验_2026-11-12" / "rebuilt"
    out.mkdir(parents=True, exist_ok=True)

    src = root / "CCMS核查_2026-10-31" / "src"
    scr = root / "CCMS核查_2026-10-31" / "scripts"
    if args.code:
        code = Path(args.code).resolve()
        sys.path.insert(0, str(code))
        deps = {n: code / n for n in ("aps_io.py", "ccms.py", "pipeline.py", "audit_metrics.py", "run_cb_firstlook.py")}
    else:
        sys.path.insert(0, str(src))
        sys.path.insert(0, str(scr))
        deps = {
            "aps_io.py": src / "aps_io.py",
            "ccms.py": src / "ccms.py",
            "pipeline.py": src / "pipeline.py",
            "audit_metrics.py": src / "audit_metrics.py",
            "run_cb_firstlook.py": scr / "run_cb_firstlook.py",
        }

    from aps_io import dev_split, load_aps_csv, split_features
    from ccms import rebuild_missing_flags
    from pipeline import MedianImputer
    from run_cb_firstlook import build_eval_inputs

    t0 = time.time()
    df, meta = load_aps_csv(str(data))
    X, y = split_features(df)
    feature_names = [c for c in df.columns if c != "class"]
    print(f"[load] X={X.shape} pos={int(y.sum())} cols={len(feature_names)} sha={meta['sha256'][:12]}", flush=True)

    # 特征顺序与数据身份
    (out / "feature_order.json").write_text(
        json.dumps(
            {
                "n_features": len(feature_names),
                "order": feature_names,
                "source_csv": str(data),
                "source_sha256": meta["sha256"],
                "note": "M3 表示的列序 = 170 列填补值 + 同序 170 列缺失标记",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    rows = []
    for s in (11, 22, 33, 44, 55):
        sp = dev_split(y, seed=s, stratify=True)
        tr, va = sp["train"], sp["val"]
        Xtr, Xva = X[tr], X[va]
        yva = y[va]

        imp = MedianImputer().fit(Xtr)               # 只 fit 训练部分
        ev, ev_meta = build_eval_inputs(Xva)["S1_30_101"]

        native = np.isnan(Xva)
        merged = np.isnan(ev)
        added = merged & ~native                      # 新增删除 = 合并后 - 原生
        assert np.array_equal(merged, native | added)

        Mv = np.hstack([imp.transform(ev), rebuild_missing_flags(ev)])
        m3_sha = sha_bytes(np.ascontiguousarray(Mv).tobytes())

        shape = np.array(Xva.shape, dtype=np.int64)
        np.savez_compressed(
            out / f"eval_masks_s{s}.npz",
            val_rows=va.astype(np.int64),
            y_val=yva.astype(np.int64),
            mask_shape=shape,
            native_bits=np.packbits(native, axis=None),
            added_bits=np.packbits(added, axis=None),
            merged_bits=np.packbits(merged, axis=None),
            n_native=np.int64(native.sum()),
            n_added=np.int64(added.sum()),
            n_merged=np.int64(merged.sum()),
            added_rate_prefix=np.array([0.30]),
            added_seed_prefix=np.array([101]),
            m3_eval_sha256=np.array([m3_sha]),
        )
        np.savez_compressed(
            out / f"imputer_s{s}.npz",
            medians=imp.medians_.astype(np.float64),
            all_nan_columns=np.array(imp.all_nan_columns_, dtype=np.int64),
            n_features_in=np.int64(imp.n_features_in_),
            fitted_on_rows=np.array([int(len(tr))]),
        )
        import joblib

        joblib.dump(imp, out / f"imputer_s{s}.joblib")

        rows.append(
            {
                "split": s,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "pos_train": int(y[tr].sum()),
                "pos_val": int(yva.sum()),
                "native_missing_cells": int(native.sum()),
                "added_cells": int(added.sum()),
                "merged_missing_cells": int(merged.sum()),
                "added_rate": 0.30,
                "added_seed": 101,
                "stress_meta_deleted": ev_meta.get("deleted"),
                "m3_eval_shape": list(Mv.shape),
                "m3_eval_sha256": m3_sha,
                "imputer_all_nan_columns": imp.all_nan_columns_,
                "imputer_median_sha256": sha_bytes(imp.medians_.astype(np.float64).tobytes()),
                "val_rows_sha256": sha_bytes(va.astype(np.int64).tobytes()),
            }
        )
        print(
            f"[rebuild] s{s}: native={native.sum()} added={added.sum()} merged={merged.sum()} "
            f"m3={Mv.shape} sha={m3_sha[:12]}",
            flush=True,
        )

    env = {
        "identity": "事后重建的复现补充材料（不是 2026-11-04 原运行落盘文件）",
        "run_id": "A3_rebuild_2026-11-12",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha_file(Path(__file__).resolve()),
        "data_sha256": meta["sha256"],
        "dependencies": {str(p): sha_file(p) for p in deps.values()},
        "recipe": {
            "split": "dev_split(y, seed=s, stratify=True), sizes=(42000, 9000, 9000)",
            "eval_input": "build_eval_inputs(Xva)['S1_30_101'] -> stress_unit(Xva, 0.30, seed=101)",
            "imputer": "MedianImputer().fit(Xtr) only; medians per column; all-NaN columns filled with 0",
            "m3": "hstack([imputer.transform(ev), isnan(ev).astype(float)]) -> (9000, 340)",
            "sample_rows": "val_rows are 0-based row indices into the original APS CSV",
        },
        "per_split": rows,
        "wall_seconds": round(time.time() - t0, 2),
    }
    (out / "rebuild_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "rebuild_summary.json").write_text(
        json.dumps({"per_split": rows, "n_splits": len(rows)}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[done] {out} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
