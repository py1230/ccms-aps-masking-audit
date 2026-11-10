"""APS 官方测试集：一次性评价（protocol v2，冻结后第一次也是唯一一次）。

规则（见 protocol_v2_冻结_2026-10-31.md 与 deviations.md）：
- α 直接取自 aps_formal 记录中各训练划分已经选出的 C* / B*，**不因测试集表现做任何调整**；
- 另外固定报告 α=0.25 与 α=1.0 两端，用于画边界；
- 训练与预处理、q 估计都只用训练集，测试集只做前向；
- 输出与主表同口径的 AP/AUROC 与配对 bootstrap 区间。

用法：python3 scripts/run_aps_testset.py [--boot 1000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aps_io import dev_split, load_aps_csv, split_features  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer, _make_model  # noqa: E402
from run_cb_firstlook import bootstrap_delta, build_eval_inputs, metrics_of  # noqa: E402

TRAIN = str(ROOT.parent / "aps_failure_training_set.csv")
TEST = str(ROOT.parent / "data" / "aps_test" / "aps_failure_test_set.csv")
MASK_SEEDS = (9107, 9108, 9109)
BOOT_CONDS = ("S0", "S1_30_101", "S2_1101", "S3_2101")
FORMAL_GLOB = "aps_formal_*.json"


def build_m3(imp, X):
    return np.hstack([imp.transform(X), rebuild_missing_flags(X)])


def pick_alphas(formal_path):
    """从正式运行记录里取每个划分已选出的 α；找不到就报错，不猜。"""
    rec = json.loads(Path(formal_path).read_text(encoding="utf-8"))
    out = {}
    for s in rec["splits"]:
        out[int(s["seed"])] = (float(s["selection"]["C"]["chosen_alpha"]),
                               float(s["selection"]["B"]["chosen_alpha"]))
    return out, rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--formal", default=None, help="默认取 records/ 下最新一份 aps_formal")
    args = ap.parse_args(argv)

    formal_path = args.formal or sorted((ROOT / "records").glob(FORMAL_GLOB))[-1]
    chosen, _ = pick_alphas(formal_path)

    t0 = time.time()
    df_tr, meta_tr = load_aps_csv(TRAIN)
    X, y = split_features(df_tr)
    df_te, meta_te = load_aps_csv(TEST)
    Xt, yt = split_features(df_te)
    if X.shape[1] != Xt.shape[1]:
        raise SystemExit(f"训练/测试特征数不一致：{X.shape[1]} vs {Xt.shape[1]}")
    evals = build_eval_inputs(Xt)

    out = {
        "tier": "formal_official_testset_single_use",
        "protocol": "protocol_v2_冻结_2026-10-31.md",
        "alpha_source": str(Path(formal_path).name),
        "note": "官方测试集第一次也是唯一一次使用；α 来自训练集选择，未按测试集表现调整",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": meta_tr, "test": {k: meta_te[k] for k in
                                   ("path", "sha256", "n_rows", "header_line_index")},
        "test_positives": int(yt.sum()), "test_positive_rate": float(yt.mean()),
        "mask_seeds": list(MASK_SEEDS), "n_bootstrap": int(args.boot),
        "splits": [],
    }
    print(f"test rows={len(yt)} pos={int(yt.sum())} rate={yt.mean():.4f} "
          f"features={Xt.shape[1]}", flush=True)

    for seed in sorted(chosen):
        ts = time.time()
        a_c, a_b = chosen[seed]
        sp = dev_split(y, seed=seed, stratify=True)
        X_tr, y_tr = X[sp["train"]], y[sp["train"]]
        imp = MedianImputer().fit(X_tr)
        q0, counts0, _ = estimate_class_rates(X_tr, y_tr)

        proba, metrics, info = {}, {}, {}

        # 基线 M3
        m = _make_model(2026)
        m.fit(build_m3(imp, X_tr), y_tr)
        proba["M3"] = {k: m.predict_proba(build_m3(imp, v[0]))[:, 1] for k, v in evals.items()}
        metrics["M3"] = {k: metrics_of(yt, proba["M3"][k]) for k in evals}
        info["M3"] = {"n_features": int(build_m3(imp, X_tr).shape[1])}

        alphas = sorted({0.25, 1.0, round(a_c, 3), round(a_b, 3)})
        for alpha in alphas:
            tag = f"{int(round(alpha * 100)):03d}"
            bvec = class_blind_rates(q0, counts0, alpha)
            ps_c, ps_b, fs = ({k: [] for k in evals}, {k: [] for k in evals}, [])
            for sd in MASK_SEEDS:
                for arm, maker, ps in (("C", None, ps_c), ("B", None, ps_b)):
                    if arm == "C":
                        Xm, _ = mask_class_conditional(X_tr, y_tr, alpha, seed=sd)
                    else:
                        Xm, _ = mask_class_blind(X_tr, bvec, seed=sd)
                    M = build_m3(imp, Xm)
                    mdl = _make_model(2026)
                    t = time.perf_counter()
                    mdl.fit(M, y_tr)
                    fs.append(time.perf_counter() - t)
                    for name, (mat, _meta) in evals.items():
                        ps[name].append(mdl.predict_proba(build_m3(imp, mat))[:, 1])
            for cfg, ps in ((f"C{tag}", ps_c), (f"B{tag}", ps_b)):
                proba[cfg] = {k: np.mean(v, axis=0) for k, v in ps.items()}
                metrics[cfg] = {k: metrics_of(yt, proba[cfg][k]) for k in evals}
                info[cfg] = {"fit_seconds_mean": float(np.mean(fs)),
                             "n_train_rows": int(X_tr.shape[0])}

        cfg_c, cfg_b = f"C{int(round(a_c * 100)):03d}", f"B{int(round(a_b * 100)):03d}"
        ci = {}
        for cname in BOOT_CONDS:
            ci[cname] = {}
            for label, a, b in ((f"C{cfg_c[1:]}-B{cfg_b[1:]}", cfg_c, cfg_b),
                                ("C-A0", cfg_c, "M3"), ("B-A0", cfg_b, "M3"),
                                ("C1.0-A0", "C100", "M3")):
                if a not in proba or b not in proba:
                    continue
                ci[cname][label] = bootstrap_delta(yt, proba[a][cname], proba[b][cname],
                                                   n_boot=args.boot)

        out["splits"].append({"seed": seed, "alpha_C": a_c, "alpha_B": a_b,
                              "chosen_cfg": {"C": cfg_c, "B": cfg_b},
                              "train_positives": int(y_tr.sum()),
                              "fit_info": info, "metrics": metrics,
                              "bootstrap": ci,
                              "elapsed_seconds": float(time.time() - ts)})
        print(f"split {seed}: aC={a_c} aB={a_b} S0 AP M3={metrics['M3']['S0']['ap']:.4f} "
              f"C*={metrics[cfg_c]['S0']['ap']:.4f} B*={metrics[cfg_b]['S0']['ap']:.4f} "
              f"C1.0={metrics['C100']['S0']['ap']:.4f} ({time.time()-ts:.0f}s)", flush=True)

    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["wall_seconds"] = float(time.time() - t0)
    path = ROOT / "records" / f"aps_testset_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
