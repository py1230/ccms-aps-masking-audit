"""CCMS（类别条件缺失率收缩）的核心算子实现 + 类别盲预算对照 B。

实现契约严格按《CCMS研究全景与修订执行计划_2026-10-31.docx》§6：

    q_jy      : 当前模型训练部分中类别 y 的第 j 列经验缺失率
    qmax_j    : max_y q_jy
    C（类别条件遮蔽）: a_jy = alpha * (qmax_j - q_jy) / (1 - q_jy)
    B（类别盲预算对照）: b_j = alpha * (qmax_j - qbar_j) / (1 - qbar_j),
                        qbar_j = sum_y pi_y * q_jy, pi_y 为训练类别比例

约定：
- 缺失用 NaN 表示；"可见单元"指原本非 NaN 的单元。
- 原本缺失的单元保持缺失，只对可见单元新增遮蔽。
- q_jy = 1（该列该类全空）没有可删单元，直接跳过，不产生除零。
- 两类 q 相同则该列不新增遮蔽。
- q 与遮蔽概率只由训练部分估计；推理与评价不重新估计、不按类别遮蔽。
- B 的预算来自训练标签统计，但逐样本遮蔽时不按类别分组（函数不接收 y）。

层级：本模块支撑实现核验（golden_evaluator_self_check），本身不产生性能证据。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "estimate_class_rates",
    "class_conditional_rates",
    "class_blind_rates",
    "mask_class_conditional",
    "mask_class_blind",
    "rebuild_missing_flags",
]


def _as_matrix(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维矩阵，实际 shape={X.shape}")
    if y is None:
        return X, None
    y = np.asarray(y).astype(int).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y 长度 {y.shape[0]} 与 X 行数 {X.shape[0]} 不一致")
    return X, y


def _qmax(q: np.ndarray) -> np.ndarray:
    """逐列取各类经验缺失率的最大值；整列无观测时记 0（该列不可能有可见单元）。"""
    if q.size == 0:
        return np.zeros(q.shape[1])
    filled = np.where(np.isnan(q), -np.inf, q)
    mx = filled.max(axis=0)
    return np.where(np.isfinite(mx), mx, 0.0)


def estimate_class_rates(X, y, n_classes=None):
    """返回 (q, counts, miss_counts)。

    q[c, j] = 类别 c 第 j 列缺失率；空类别的 q 记为 NaN（该类别没有样本，不参与 qmax）。
    """
    X, y = _as_matrix(X, y)
    n_cols = X.shape[1]
    if n_classes is None:
        n_classes = int(y.max()) + 1 if y.size else 0
    n_classes = int(n_classes)
    q = np.full((n_classes, n_cols), np.nan, dtype=float)
    counts = np.zeros(n_classes, dtype=int)
    miss_counts = np.zeros((n_classes, n_cols), dtype=int)
    miss = np.isnan(X)
    for c in range(n_classes):
        sel = y == c
        counts[c] = int(sel.sum())
        if counts[c] > 0:
            miss_counts[c] = miss[sel].sum(axis=0)
            q[c] = miss_counts[c] / counts[c]
    return q, counts, miss_counts


def class_conditional_rates(q, alpha):
    """C 的新增遮蔽概率 a[c, j] = alpha*(qmax_j - q_cj)/(1 - q_cj)。"""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须落在 [0, 1]，实际 {alpha}")
    qmax = _qmax(q)
    a = np.zeros_like(q)
    denom = 1.0 - q
    ok = (denom > 0) & ~np.isnan(q)
    a[ok] = alpha * (qmax[None, :] - q)[ok] / denom[ok]
    return np.clip(a, 0.0, 1.0)


def class_blind_rates(q, counts, alpha):
    """B 的逐列新增遮蔽概率 b_j = alpha*(qmax_j - qbar_j)/(1 - qbar_j)。"""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    counts = np.asarray(counts, dtype=float).ravel()
    if q.shape[0] != counts.shape[0]:
        raise ValueError("q 的类别数与 counts 长度不一致")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须落在 [0, 1]，实际 {alpha}")
    total = counts.sum()
    if total <= 0:
        return np.zeros(q.shape[1])
    pi = counts / total
    q_filled = np.where(np.isnan(q), 0.0, q)
    qbar = pi @ q_filled
    qmax = _qmax(q)
    b = np.zeros(q.shape[1], dtype=float)
    denom = 1.0 - qbar
    ok = denom > 0
    b[ok] = alpha * (qmax[ok] - qbar[ok]) / denom[ok]
    return np.clip(b, 0.0, 1.0)


def _draw(uniforms, shape, seed):
    if uniforms is not None:
        U = np.asarray(uniforms, dtype=float)
        if U.shape != shape:
            raise ValueError(f"uniforms shape {U.shape} 与 X shape {shape} 不一致")
        return U
    return np.random.default_rng(seed).random(shape)


def _deletion_summary(new_mask, y, n_classes, q, counts, alpha, expected):
    per_class = np.zeros((n_classes, new_mask.shape[1]), dtype=int)
    if y is not None and y.size:
        for c in range(n_classes):
            sel = y == c
            if sel.any():
                per_class[c] = new_mask[sel].sum(axis=0)
    return {
        "alpha": float(alpha),
        "q_before": q,
        "counts": counts,
        "new_mask": new_mask,
        "deleted_per_col": new_mask.sum(axis=0),
        "deleted_per_class_col": per_class,
        "deleted_total": int(new_mask.sum()),
        "expected_deleted_per_col": expected,
    }


def mask_class_conditional(X, y, alpha, seed=None, uniforms=None, n_classes=None):
    """按 C 对可见单元新增遮蔽，返回 (X_new, info)。原始输入不被修改。"""
    X, y = _as_matrix(X, y)
    q, counts, _ = estimate_class_rates(X, y, n_classes)
    n_classes = q.shape[0]
    a = class_conditional_rates(q, alpha)
    if y.size == 0:
        info = _deletion_summary(np.zeros(X.shape, dtype=bool), y, n_classes, q, counts, alpha,
                                 np.zeros(X.shape[1]))
        return X.copy(), info
    U = _draw(uniforms, X.shape, seed)
    visible = ~np.isnan(X)
    new_mask = visible & (U < a[y])
    X_new = X.copy()
    X_new[new_mask] = np.nan
    qmax = _qmax(q)
    expected = alpha * (counts[:, None] * (qmax[None, :] - np.where(np.isnan(q), 0.0, q))).sum(axis=0)
    info = _deletion_summary(new_mask, y, n_classes, q, counts, alpha, expected)
    info["a"] = a
    info["qmax"] = qmax
    info["seed"] = seed
    return X_new, info


def mask_class_blind(X, b, seed=None, uniforms=None):
    """按 B 对可见单元新增遮蔽。函数不接收标签，逐样本不按类别分组。"""
    X, _ = _as_matrix(X)
    b = np.asarray(b, dtype=float).ravel()
    if b.shape[0] != X.shape[1]:
        raise ValueError(f"b 长度 {b.shape[0]} 与列数 {X.shape[1]} 不一致")
    U = _draw(uniforms, X.shape, seed)
    visible = ~np.isnan(X)
    new_mask = visible & (U < b[None, :])
    X_new = X.copy()
    X_new[new_mask] = np.nan
    info = {
        "new_mask": new_mask,
        "b": b,
        "deleted_per_col": new_mask.sum(axis=0),
        "deleted_total": int(new_mask.sum()),
        "seed": seed,
    }
    return X_new, info


def rebuild_missing_flags(X):
    """遮蔽后按新矩阵重建全部列的缺失标记（1=缺失）。"""
    return np.isnan(np.asarray(X, dtype=float)).astype(float)
