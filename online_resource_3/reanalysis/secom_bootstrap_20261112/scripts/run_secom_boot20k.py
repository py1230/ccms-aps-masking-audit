"""SECOM 折内分层 bootstrap：本次重跑 20,000 次。

identity: 本次重跑（2026-11-12），不是 2026-11-08 那批 20,000 次的原件。
口径来源：rr_副本/02_原完整复现包/…/scripts/summarize.py::secom_summary（只读，未修改）。
实现策略：prep / wap / interval 三个函数按原脚本逐字复制，保证语义一致；
随机流同样为每折 default_rng(20261101 + i)，因此前 1,000 次 draw 与原 1,000 次运行使用同一条随机流。
校验门：比较前 1,000 次 draw 的**汇总值**（delta_mean、ci95 端点、五折点估计）是否与归档 secom_summary.json 相等；不相等则以非零码退出，不产出 20,000 次结果。
边界：归档没有保存逐次差值数组，因此这里能证明的是「汇总值相同」；不能声称与历史逐次数组逐位相同。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP

ROOT = Path("/home/user/workspace/默认项目/data-analysis/secom_bootstrap_20k_2026-11-12")
INPUT = Path("/home/user/workspace/默认项目/原件补件_2026-11-11/02_SECOM_bootstrap与重采样矩阵")
OUT = ROOT / "exports"
LOGS = ROOT / "logs"

N_BOOT = 20000
N_VALID = 1000
BASE_SEED = 20261101
FAM = {"S0": ["S0"], "S1_30": ["S1_30_101", "S1_30_202", "S1_30_303"]}
CELLS = [("fixed_C-B", "C025", "B025"), ("selected_C-B", "C_selected", "B_selected")]
TOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- 以下两个函数与 summarize.py 逐字一致 ----------
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
    return float(
        np.sum(np.diff(np.r_[0.0, tp]) * np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)) / total
    )


def interval(v):
    return [float(z) for z in np.quantile(v, [0.025, 0.975])]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    packs = [np.load(INPUT / f"secom_fold{i}.npz") for i in range(5)]
    arch = json.loads((INPUT / "secom_summary.json").read_text(encoding="utf-8"))

    # 冒烟自检：均匀权重下 wap 必须等于 sklearn 的 AP
    rng = np.random.default_rng(7)
    y_s = np.r_[1, np.zeros(20), 1, np.zeros(11)]
    p_s = rng.integers(0, 4, len(y_s)) / 4.0  # 故意制造并列
    w_s = np.ones(len(y_s))
    d_smoke = abs(wap(prep(y_s, p_s), w_s) - AP(y_s, p_s))
    print(f"[smoke] |wap - sklearn AP| = {d_smoke:.3e}", flush=True)
    if d_smoke > 1e-12:
        print("[FATAL] wap 与 AP 不一致，停止", flush=True)
        return 2

    # ---------- 第 1 步：生成并保存逐次重采样计数矩阵 ----------
    counts = []
    for i, z in enumerate(packs):
        y = z["y_test"]
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        rng = np.random.default_rng(BASE_SEED + i)
        c = np.empty((N_BOOT, len(y)), dtype=np.uint16)
        for b in range(N_BOOT):
            ids = np.r_[
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ]
            c[b] = np.bincount(ids, minlength=len(y)).astype(np.uint16)
            if b % 5000 == 0:
                print(f"[counts] fold{i} draw {b}/{N_BOOT}", flush=True)
        assert c.sum(axis=1).min() == len(y) and c.sum(axis=1).max() == len(y)
        assert int(c.max()) < 65535
        # 正类计数必须全部 > 0，否则 AP 无定义
        assert int((c[:, pos].sum(axis=1) == 0).sum()) == 0
        counts.append(c)
        print(f"[counts] fold{i} done, n={len(y)}, pos={len(pos)}, max_count={int(c.max())}", flush=True)

    np.savez_compressed(
        OUT / "secom_boot20k_resample_counts.npz",
        **{f"fold{i}": counts[i] for i in range(5)},
        n_boot=np.int64(N_BOOT),
        base_seed=np.int64(BASE_SEED),
        schema=np.array(["row = draw index; col = test-row index within fold; value = times drawn"]),
    )
    print(f"[counts] saved ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 第 2 步：四格逐次 ΔAP ----------
    draws = {}
    summary = {}
    validation = {"identity": "re-run validation vs archived 1000-draw summary", "cells": {}}

    for label, ca, cb in CELLS:
        summary[label] = {}
        validation["cells"][label] = {}
        for fam, ks in FAM.items():
            d = np.empty((N_BOOT, 5), dtype=np.float64)
            points = []
            for i, z in enumerate(packs):
                y = z["y_test"]
                pa = [z[f"{ca}__test__{k}"] for k in ks]
                pb = [z[f"{cb}__test__{k}"] for k in ks]
                prs = [(prep(y, a), prep(y, b)) for a, b in zip(pa, pb)]
                points.append(float(np.mean([AP(y, a) - AP(y, b) for a, b in zip(pa, pb)])))
                c = counts[i]
                for b in range(N_BOOT):
                    w = c[b].astype(np.float64)
                    d[b, i] = np.mean([wap(a, w) - wap(bb, w) for a, bb in prs])
                print(f"[draws] {label} {fam} fold{i} done ({time.time()-t0:.0f}s)", flush=True)

            dm = d.mean(axis=1)
            rec = {
                "delta_mean": float(np.mean(points)),
                "ci95": interval(dm),
                "per_fold": points,
                "per_fold_ci95": [interval(d[:, i]) for i in range(5)],
                "draw_mean_sd": float(np.std(dm, ddof=1)),
                "draw_mean_mc_se": float(np.std(dm, ddof=1) / np.sqrt(N_BOOT)),
                "draw_mean_mc_se_note": "Monte Carlo standard error of the bootstrap mean (SD/sqrt(n_boot)); not the sampling SE of the AP difference",
                "draw_mean_of_draw_means": float(dm.mean()),
                "percentiles": {str(q): float(np.quantile(dm, q / 100)) for q in (1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)},
                "n_draws_negative": int((dm < 0).sum()),
                "n_draws_positive": int((dm > 0).sum()),
                "n_draws_zero": int((dm == 0).sum()),
                "frac_draws_negative": float((dm < 0).mean()),
                "min": float(dm.min()),
                "max": float(dm.max()),
                "n_boot": int(N_BOOT),
            }
            summary[label][fam] = rec
            draws[f"{label}__{fam}"] = d

            # ---- 校验门：前 1,000 次 vs 归档 ----
            a = arch["paired"][label][fam]
            first = dm[:N_VALID]
            v = {
                "archived_delta_mean": a["delta_mean"],
                "rerun_delta_mean": rec["delta_mean"],
                "delta_mean_abs_diff": abs(rec["delta_mean"] - a["delta_mean"]),
                "archived_ci95": a["ci95"],
                "rerun_ci95_first1000": interval(first),
                "ci95_abs_diff": [abs(x - y2) for x, y2 in zip(interval(first), a["ci95"])],
                "archived_per_fold": a["per_fold"],
                "rerun_per_fold": points,
                "per_fold_abs_diff": [abs(x - y2) for x, y2 in zip(points, a["per_fold"])],
            }
            v["pass"] = bool(
                v["delta_mean_abs_diff"] <= TOL
                and max(v["ci95_abs_diff"]) <= TOL
                and max(v["per_fold_abs_diff"]) <= TOL
            )
            validation["cells"][label][fam] = v
            print(
                f"[validate] {label} {fam}: dmean_diff={v['delta_mean_abs_diff']:.2e} "
                f"ci_diff={max(v['ci95_abs_diff']):.2e} pass={v['pass']}",
                flush=True,
            )

    all_pass = all(c[f]["pass"] for c in validation["cells"].values() for f in c)
    validation["all_pass"] = all_pass
    (OUT / "secom_boot20k_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if not all_pass:
        print("[FATAL] 1,000 次校验未通过，丢弃 20,000 次结果", flush=True)
        return 3

    np.savez_compressed(
        OUT / "secom_boot20k_draws.npz",
        **draws,
        n_boot=np.int64(N_BOOT),
        base_seed=np.int64(BASE_SEED),
        schema=np.array(["shape (n_boot, 5) = (draw, outer fold); value = delta AP of that draw in that fold"]),
    )

    payload = {
        "identity": "本次重跑（re-run, 2026-11-12）；不是 2026-11-08 那批 20,000 次的原件",
        "n_bootstrap": N_BOOT,
        "base_seed": BASE_SEED,
        "rng": "per fold np.random.default_rng(20261101 + fold_index), sequential draws",
        "resample": "within outer fold and label (pos/neg resampled with replacement, size preserved); weights = draw counts; weighted AP",
        "estimand": "mean over folds of [mean over condition replicas of AP(C)-AP(B)]; ci95 = 2.5/97.5 percentile of the per-draw mean over folds",
        "ci95_definition": "percentile interval of the 20,000 per-draw cross-fold means; resampling interval, not a significance test",
        "notes": [
            "delta_mean 由五折点估计取均值，与重采样次数无关。",
            "draw_mean_mc_se 是 bootstrap 均值的 Monte Carlo 标准误，不可当作 AP 差的抽样标准误或区间精度。",
            "与归档 secom_summary.json 的一致性只覆盖汇总值（点估计、区间端点、逐折点估计）；归档未保存逐次数组。",
        ],
        "family": {k: v for k, v in FAM.items()},
        "cells": summary,
        "validation_vs_archived_1000": "see secom_boot20k_validation.json (all_pass=%s)" % all_pass,
        "archived_source": str(INPUT / "secom_summary.json"),
        "inputs": {str(INPUT / f"secom_fold{i}.npz"): sha256(INPUT / f"secom_fold{i}.npz") for i in range(5)},
        "wall_seconds": float(time.time() - t0),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "secom_boot20k_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    env = {
        "identity": "本次重跑（re-run, 2026-11-12）；不是 2026-11-08 那批 20,000 次的原件，原件在云电脑上不存在",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "numpy_random_note": "随机流只在本文件记录的 Python/numpy 版本内复现；跨 numpy 版本未测试，不作跨版本逐位一致保证",
        "command": "python3 scripts/run_secom_boot20k.py（由 scripts/run_with_receipt.sh 调用）",
        "cwd": str(Path.cwd()),
        "params": {"n_boot": N_BOOT, "n_validate": N_VALID, "base_seed": BASE_SEED, "cells": [c[0] for c in CELLS]},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "secom_boot20k_environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("\n===== 结果（本次重跑，20,000 次）=====", flush=True)
    for label in summary:
        for fam in summary[label]:
            r = summary[label][fam]
            print(
                f"{label:14s} {fam:6s} delta={r['delta_mean']:+.6f} "
                f"ci95=[{r['ci95'][0]:+.6f},{r['ci95'][1]:+.6f}] "
                f"P(<0)={r['frac_draws_negative']:.4f}",
                flush=True,
            )
    print(f"\nALL_DONE wall={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
