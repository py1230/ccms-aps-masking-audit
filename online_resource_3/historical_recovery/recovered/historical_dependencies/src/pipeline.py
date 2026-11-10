"""M2 / M3 / M4 的预处理与训练管线（防泄漏版）。

按计划文档 §6/§8：
    M2  仅填补 + HGB
    M3  填补 + 全列缺失标记 + HGB
    M4  保留 NaN 的原生 HGB（不做填补、不加标记）

纪律：
- 填补器只在「当前模型训练部分」拟合，拟合后固定，各配置共用；
- 缺失标记一律从原始（遮蔽后、填补前）矩阵重建，不从填补后的矩阵倒推；
- 评价部分只做 transform，不参与任何 fit；
- 超参数文档未规定，这里只用 scikit-learn 默认值并显式记录，冻结协议时必须确认。

本模块不产生任何性能结论；它只保证「算得对、没泄漏、可复现」。
"""
from __future__ import annotations

import warnings

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from audit_metrics import evaluate_all

__all__ = ["MedianImputer", "build_matrix", "CONFIGS", "fit_predict", "missing_flags"]

CONFIGS = {
    "M2": {"impute": True, "add_flags": False, "keep_nan": False},
    "M3": {"impute": True, "add_flags": True, "keep_nan": False},
    "M4": {"impute": False, "add_flags": False, "keep_nan": True},
}


def missing_flags(X):
    """从原始矩阵重建全列缺失标记（1=缺失）。"""
    return np.isnan(np.asarray(X, dtype=float)).astype(float)


class MedianImputer:
    """逐列中位数填补器；全空列用 0 填补并记录列号。"""

    def __init__(self):
        self.medians_ = None
        self.all_nan_columns_ = []
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 必须是二维，实际 {X.shape}")
        self.n_features_in_ = X.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        all_nan = np.flatnonzero(~np.isfinite(med))
        self.all_nan_columns_ = [int(i) for i in all_nan]
        med = np.where(np.isfinite(med), med, 0.0)
        self.medians_ = med
        return self

    def transform(self, X):
        if self.medians_ is None:
            raise RuntimeError("先 fit 再 transform；评价部分不得自己拟合填补器")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"列数不一致：fit 时 {self.n_features_in_}，当前 {X.shape[1]}")
        out = X.copy()
        rows, cols = np.where(np.isnan(out))
        out[rows, cols] = self.medians_[cols]
        return out


def build_matrix(X, imputer, add_flags=False, keep_nan=False):
    """按配置拼出模型输入矩阵。keep_nan=True 时原样返回（交给原生支持 NaN 的模型）。"""
    X = np.asarray(X, dtype=float)
    if keep_nan:
        if add_flags:
            raise ValueError("保留 NaN 的原生模型不需要再叠标记，二者择一")
        return X.copy()
    if imputer is None:
        raise ValueError("非 keep_nan 配置必须提供已拟合的填补器")
    filled = imputer.transform(X)
    if not add_flags:
        return filled
    return np.hstack([filled, missing_flags(X)])


def _make_model(seed):
    """文档未规定超参数：使用 sklearn 默认值 + 固定 random_state，并记录全部参数。"""
    return HistGradientBoostingClassifier(random_state=int(seed))


def fit_predict(config, X_train, y_train, X_eval, y_eval=None, seed=2026):
    if config not in CONFIGS:
        raise ValueError(f"未知配置 {config!r}，可用：{sorted(CONFIGS)}")
    spec = CONFIGS[config]
    X_train = np.asarray(X_train, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y_train = np.asarray(y_train).ravel().astype(int)
    if X_train.shape[1] != X_eval.shape[1]:
        raise ValueError("训练与评价的列数不一致")
    if len(y_train) != len(X_train):
        raise ValueError("y 与 X 行数不一致")

    imputer = MedianImputer().fit(X_train) if spec["impute"] else None
    M_train = build_matrix(X_train, imputer, spec["add_flags"], spec["keep_nan"])
    M_eval = build_matrix(X_eval, imputer, spec["add_flags"], spec["keep_nan"])

    model = _make_model(seed)
    model.fit(M_train, y_train)
    proba = model.predict_proba(M_eval)[:, 1]
    params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in model.get_params().items()}

    metrics = None
    if y_eval is not None:
        metrics = evaluate_all(np.asarray(y_eval).ravel().astype(int), proba)

    record = {
        "config": config,
        "spec": dict(spec),
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
        "positives_train": int(y_train.sum()),
        "n_features_input": int(X_train.shape[1]),
        "n_features_matrix": int(M_train.shape[1]),
        "hyperparameters": params,
        "imputer_all_nan_columns": list(imputer.all_nan_columns_) if imputer else [],
        "sklearn_version": sklearn.__version__,
        "note": "超参数未在计划文档中规定，此处为 sklearn 默认值 + 固定 random_state，冻结协议时需确认",
    }
    return {
        "proba": proba,
        "metrics": metrics,
        "record": record,
        "model": model,
        "imputer_medians": imputer.medians_ if imputer is not None else None,
    }
