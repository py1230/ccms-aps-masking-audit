"""受控模拟：当类间缺失差是「采集痕迹」且在部署时改变，C 会不会占优（协议 P2）。

生成过程与场景见 protocol/探索性协议_P2_受控模拟_2026-10-31.md。
实现在跑之前写定；所有随机种子、缺失率、场景都在记录里可复原。

用法：python3 scripts/sim_nuisance.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_metrics import average_precision, auroc  # noqa: E402
from ccms import (  # noqa: E402
    class_blind_rates,
    estimate_class_rates,
    mask_class_blind,
    mask_class_conditional,
    rebuild_missing_flags,
)
from pipeline import MedianImputer, _make_model  # noqa: E402

P = 20
GAP_COLS = 6
B_GAP = 0.20
DELTA = 0.40
PI_TARGET = 0.05
N = 8000
N_TRAIN = 6400
DATA_SEEDS = (501, 502, 503, 504, 505)
MASK_SEEDS = (11, 12, 13)
ALPHAS = (0.25, 0.5, 1.0)
MODEL_SEED = 2026
SCENARIOS = ("S0_natural", "S1_extra10", "Shift0_nogap", "ShiftN_reversed")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gen_data(seed):
    """特征 → 潜分数 → 标签；截距用二分法把正类比例调到 0.05。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P))
    s = 1.2 * (X[:, 0] + X[:, 1] + X[:, 2])
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sigmoid(s - mid).mean() > PI_TARGET:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    y = (rng.random(N) < _sigmoid(s - c)).astype(int)
    return X, y, {"intercept": float(c), "positives": int(y.sum()),
                  "prevalence": float(y.mean())}


def base_rates(seed):
    rng = np.random.default_rng(seed + 9000)
    b = rng.uniform(0.05, 0.30, size=P)
    b[:GAP_COLS] = B_GAP
    delta = np.zeros(P)
    delta[:GAP_COLS] = DELTA
    return b, delta


def apply_missing(X, y, b, delta, seed, pi=PI_TARGET):
    """给定类别后 MCAR：q_j1 = b_j + (1−pi)δ_j，q_j0 = b_j − pi·δ_j。"""
    rng = np.random.default_rng(seed)
    q1 = np.clip(b + (1 - pi) * delta, 0.0, 1.0)
    q0 = np.clip(b - pi * delta, 0.0, 1.0)
    Q = np.where(y[:, None] == 1, q1[None, :], q0[None, :])
    M = rng.random(X.shape) < Q
    Xm = X.copy()
    Xm[M] = np.nan
    return Xm, {"q1_mean": float(q1.mean()), "q0_mean": float(q0.mean()),
                "gap_mean": float((q1 - q0).mean()), "missing_rate": float(M.mean())}


def extra_unit(X, rate, seed):
    U = np.random.default_rng(seed).random(X.shape)
    new = (~np.isnan(X)) & (U < rate)
    Xn = X.copy()
    Xn[new] = np.nan
    return Xn


def run_config(X_train_used, y_train, eval_sets, y_eval):
    """一次拟合，四种场景各评一次；返回 {场景: (AP, AUROC)}。"""
    imp = MedianImputer().fit(X_train_used)
    M_tr = np.hstack([imp.transform(X_train_used), rebuild_missing_flags(X_train_used)])
    model = _make_model(MODEL_SEED)
    model.fit(M_tr, y_train)
    out = {}
    for name, Xev in eval_sets.items():
        M_ev = np.hstack([imp.transform(Xev), rebuild_missing_flags(Xev)])
        p = model.predict_proba(M_ev)[:, 1]
        out[name] = (float(average_precision(y_eval, p)), float(auroc(y_eval, p)))
    return out, int(M_tr.shape[1])


def main(argv=None):
    t0 = time.time()
    out = {"protocol": "探索性协议_P2_受控模拟_2026-10-31.md",
           "run_tier": "exploratory_simulation",
           "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "params": {"P": P, "GAP_COLS": GAP_COLS, "B_GAP": B_GAP, "DELTA": DELTA,
                      "PI_TARGET": PI_TARGET, "N": N, "N_TRAIN": N_TRAIN,
                      "data_seeds": list(DATA_SEEDS), "mask_seeds": list(MASK_SEEDS),
                      "alphas": list(ALPHAS), "model_seed": MODEL_SEED},
           "per_seed": []}

    for ds in DATA_SEEDS:
        X, y, meta = gen_data(ds)
        rng = np.random.default_rng(ds + 7)
        idx = rng.permutation(N)
        tr, ev = idx[:N_TRAIN], idx[N_TRAIN:]
        b, delta = base_rates(ds)

        X_tr_m, m_tr = apply_missing(X[tr], y[tr], b, delta, seed=ds * 10 + 1)
        X_ev_nat, m_ev = apply_missing(X[ev], y[ev], b, delta, seed=ds * 10 + 2)
        eval_sets = {
            "S0_natural": X_ev_nat,
            "S1_extra10": extra_unit(X_ev_nat, 0.10, seed=ds * 10 + 3),
            "Shift0_nogap": apply_missing(X[ev], y[ev], b, np.zeros(P), seed=ds * 10 + 4)[0],
            "ShiftN_reversed": apply_missing(X[ev], y[ev], b, -delta, seed=ds * 10 + 5)[0],
        }
        rec = {"data_seed": ds, "meta": meta, "missing": {"train": m_tr, "eval": m_ev},
               "runs": []}

        per, nfeat = run_config(X_tr_m, y[tr], eval_sets, y[ev])
        rec["runs"].append({"key": "A0", "method": "A0", "alpha": 0.0, "mask_seed": None,
                            "n_features": nfeat,
                            "ap": {k: v[0] for k, v in per.items()},
                            "auroc": {k: v[1] for k, v in per.items()}})

        q0, counts0, _ = estimate_class_rates(X_tr_m, y[tr])
        for alpha in ALPHAS:
            bvec = class_blind_rates(q0, counts0, alpha)
            for ms in MASK_SEEDS:
                Xc, info_c = mask_class_conditional(X_tr_m, y[tr], alpha, seed=ms)
                Xb, info_b = mask_class_blind(X_tr_m, bvec, seed=ms)
                for meth, Xuse, info in (("C", Xc, info_c), ("B", Xb, info_b)):
                    per, nfeat = run_config(Xuse, y[tr], eval_sets, y[ev])
                    rec["runs"].append({
                        "key": f"{meth}{int(round(alpha * 100)):03d}", "method": meth,
                        "alpha": alpha, "mask_seed": ms, "n_features": nfeat,
                        "deleted_total": int(info["deleted_total"]),
                        "ap": {k: v[0] for k, v in per.items()},
                        "auroc": {k: v[1] for k, v in per.items()},
                    })
        out["per_seed"].append(rec)
        a0 = rec["runs"][0]["ap"]
        print(f"seed {ds}: A0 S0={a0['S0_natural']:.4f} Shift0={a0['Shift0_nogap']:.4f} "
              f"ShiftN={a0['ShiftN_reversed']:.4f}  (正类 {meta['positives']}, "
              f"训练缺失 {m_tr['missing_rate']:.3f}, 类差 {m_tr['gap_mean']:.3f})")

    # 汇总：每个 (方法, α, 场景) 跨 5 个数据种子
    summary = {}
    for meth in ("A0", "B", "C"):
        for alpha in ([0.0] if meth == "A0" else ALPHAS):
            key = f"{meth}_{alpha}"
            vals = {}
            for name in SCENARIOS:
                arr = []
                for s in out["per_seed"]:
                    runs = [r for r in s["runs"] if r["method"] == meth and r["alpha"] == alpha]
                    arr.append(float(np.mean([r["ap"][name] for r in runs])))
                vals[name] = {"mean": float(np.mean(arr)), "min": float(np.min(arr)),
                              "max": float(np.max(arr)), "per_seed": arr}
            summary[key] = vals
    out["summary"] = summary
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["wall_seconds"] = float(time.time() - t0)

    path = ROOT / "records" / f"sim_nuisance_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n场景               方法  α      平均AP   最小     最大")
    for name in SCENARIOS:
        for meth in ("A0", "B", "C"):
            for alpha in ([0.0] if meth == "A0" else ALPHAS):
                v = summary[f"{meth}_{alpha}"][name]
                print(f"{name:<19}{meth:<5}{alpha:<7}{v['mean']:.4f}  {v['min']:.4f}  {v['max']:.4f}")
        print()
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
