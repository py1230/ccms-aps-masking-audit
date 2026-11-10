"""Class-conditional masking C and its expected-column-budget comparator B.

Estimate class-specific missingness rates from the current training partition.
C deletes visible cells with alpha*(qmax-q_class)/(1-q_class).
B uses alpha*(qmax-qbar)/(1-qbar), where qbar weights training classes.
B uses training labels to calibrate its column budget; allocation within a
column does not use labels. Original NaNs are preserved. A class-column
with q=1 has no visible cells and is skipped. Missingness indicators must
be rebuilt after masking. C and B share base uniforms when passed the same
seed or explicit uniforms; this matches expectations, not realized counts.
This module alone provides no predictive or historical provenance evidence.
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
        raise ValueError(f"X must be two-dimensional; shape={X.shape}")
    if y is None:
        return X, None
    y = np.asarray(y).astype(int).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y length {y.shape[0]} differs from X rows {X.shape[0]} is inconsistent")
    return X, y


def _qmax(q: np.ndarray) -> np.ndarray:
    """Columnwise maximum class missingness; empty/undefined columns use zero."""
    if q.size == 0:
        return np.zeros(q.shape[1])
    filled = np.where(np.isnan(q), -np.inf, q)
    mx = filled.max(axis=0)
    return np.where(np.isfinite(mx), mx, 0.0)


def estimate_class_rates(X, y, n_classes=None):
    """Return class missingness rates q, class counts and missing-cell counts. Empty classes have NaN rates and do not contribute to qmax."""
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
    """Return C deletion probabilities alpha*(qmax-q)/(1-q) for visible cells."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; received {alpha}")
    qmax = _qmax(q)
    a = np.zeros_like(q)
    denom = 1.0 - q
    ok = (denom > 0) & ~np.isnan(q)
    a[ok] = alpha * (qmax[None, :] - q)[ok] / denom[ok]
    return np.clip(a, 0.0, 1.0)


def class_blind_rates(q, counts, alpha):
    """Return B probabilities alpha*(qmax-qbar)/(1-qbar), calibrated using training labels."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    counts = np.asarray(counts, dtype=float).ravel()
    if q.shape[0] != counts.shape[0]:
        raise ValueError("q class count differs from counts length")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; received {alpha}")
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
            raise ValueError(f"uniforms shape {U.shape} differs from X shape {shape} is inconsistent")
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
    """Apply C to visible cells and return a copied matrix and mask diagnostics."""
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
    """Apply B to visible cells. This allocation function accepts no labels."""
    X, _ = _as_matrix(X)
    b = np.asarray(b, dtype=float).ravel()
    if b.shape[0] != X.shape[1]:
        raise ValueError(f"b length {b.shape[0]} differs from number of columns {X.shape[1]} is inconsistent")
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
    """Rebuild missingness indicators after masking; 1 denotes missing."""
    return np.isnan(np.asarray(X, dtype=float)).astype(float)
