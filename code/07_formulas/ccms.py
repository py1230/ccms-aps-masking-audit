"""Core implementation of CCMS (class-conditional missingness-rate contraction) and of the class-blind budget-matched control B.

The implementation contract follows section 6 of the CCMS research plan and revision schedule (2026-10-31):

    q_jy      : empirical missingness rate of class y in column j within the current training part
    qmax_j    : max_y q_jy
    C (class-conditional masking): a_jy = alpha * (qmax_j - q_jy) / (1 - q_jy)
    B (class-blind budget-matched control): b_j = alpha * (qmax_j - qbar_j) / (1 - qbar_j),
                        qbar_j = sum_y pi_y * q_jy, where pi_y is the training class proportion

Conventions:
- Missing values are represented by NaN; a "visible cell" is a cell that is not NaN in the input.
- Cells that are already missing stay missing; masking is added to visible cells only.
- q_jy = 1 (the column is empty for that class) leaves no deletable cell and is skipped, so no division by zero occurs.
- If both classes have the same q, no masking is added in that column.
- q and the masking probabilities are estimated on the training part only; inference and evaluation neither re-estimate them nor mask by class.
- B derives its budget from training-label statistics but does not group cells by class when masking (the function does not receive y).

Level: this module supports implementation verification (golden_evaluator_self_check) and produces no performance evidence by itself.
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
        raise ValueError(f"X must be a 2-D matrix, got shape={X.shape}")
    if y is None:
        return X, None
    y = np.asarray(y).astype(int).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"length of y ({y.shape[0]}) does not match the number of rows in X ({X.shape[0]})")
    return X, y


def _qmax(q: np.ndarray) -> np.ndarray:
    """Column-wise maximum of the class-specific empirical missingness rates; a column with no observation is recorded as 0 (it cannot contain visible cells)."""
    if q.size == 0:
        return np.zeros(q.shape[1])
    filled = np.where(np.isnan(q), -np.inf, q)
    mx = filled.max(axis=0)
    return np.where(np.isfinite(mx), mx, 0.0)


def estimate_class_rates(X, y, n_classes=None):
    """Return (q, counts, miss_counts).

    q[c, j] = missingness rate of class c in column j; an empty class is recorded as NaN (no samples, so it does not enter qmax).
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
    """Additional masking probability of C, a[c, j] = alpha*(qmax_j - q_cj)/(1 - q_cj)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    qmax = _qmax(q)
    a = np.zeros_like(q)
    denom = 1.0 - q
    ok = (denom > 0) & ~np.isnan(q)
    a[ok] = alpha * (qmax[None, :] - q)[ok] / denom[ok]
    return np.clip(a, 0.0, 1.0)


def class_blind_rates(q, counts, alpha):
    """Column-wise additional masking probability of B, b_j = alpha*(qmax_j - qbar_j)/(1 - qbar_j)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    counts = np.asarray(counts, dtype=float).ravel()
    if q.shape[0] != counts.shape[0]:
        raise ValueError("the number of classes in q does not match the length of counts")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
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
            raise ValueError(f"uniforms shape {U.shape} does not match X shape {shape}")
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
    """Add masking to visible cells under C and return (X_new, info). The input array is not modified."""
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
    """Add masking to visible cells under B. The function takes no labels and does not group cells by class."""
    X, _ = _as_matrix(X)
    b = np.asarray(b, dtype=float).ravel()
    if b.shape[0] != X.shape[1]:
        raise ValueError(f"length of b ({b.shape[0]}) does not match the number of columns ({X.shape[1]})")
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
    """Rebuild the missingness indicators of all columns from the new matrix (1 = missing)."""
    return np.isnan(np.asarray(X, dtype=float)).astype(float)
