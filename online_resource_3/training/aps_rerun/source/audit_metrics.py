"""从定义重算分类指标，用于独立核对既有运行记录。

设计原则：这里不调用原训练脚本的任何函数，也不读它打印的指标；
AP 按阈值分组后的精度-召回积分计算（与并列分数处理一致），AUROC 用
Mann-Whitney 秩统计（并列取平均秩）。sklearn 只在测试里作交叉核对，
不作为定义来源。

未定义的情形返回 NaN，不返回 0：没有正类时 AP 未定义；没有预测正类时
precision 未定义。把未定义写成 0 会让「漏检」看起来像「算错了」。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "average_precision",
    "auroc",
    "confusion_at",
    "precision_recall_f1_at",
    "evaluate_all",
]


def _prep(y, scores):
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y 与 scores 长度不一致：{y.shape} vs {s.shape}")
    if np.isnan(s).any():
        raise ValueError("scores 含 NaN，指标无定义")
    return y, s


def average_precision(y, scores):
    """AP = Σ_n (R_n − R_{n−1})·P_n，阈值按不同分数分组（并列分数同一阈值）。"""
    y, s = _prep(y, scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    # 每个不同分数为一组，取组末尾的累计计数
    last_in_group = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_in_group].astype(float)
    fp = (last_in_group + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def _average_ranks(x):
    """升序平均秩（并列取平均），不依赖 scipy。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(y, scores):
    y, s = _prep(y, scores)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_at(y, scores, threshold=0.5):
    """阈值判正类用 >=（与常见 0.5 约定一致，并在报告里写明）。"""
    y, s = _prep(y, scores)
    pred = (s >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(threshold),
            "rule": "score >= threshold"}


def precision_recall_f1_at(y, scores, threshold=0.5):
    c = confusion_at(y, scores, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1,
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": c["tn"]}


def evaluate_all(y, scores, thresholds=(0.5,)):
    y, s = _prep(y, scores)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "ap": average_precision(y, s),
        "auroc": auroc(y, s),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }
    out["at_threshold"] = {str(t): precision_recall_f1_at(y, s, t) for t in thresholds}
    return out
