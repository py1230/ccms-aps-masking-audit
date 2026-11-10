"""APS 读入、缺失画像与开发内划分。

要点（来自计划文档 §7）：
- APS 的 CSV 前面有说明段落，表头不在第一行；按「某一行被逗号切开后含 class 字段」
  定位表头，并把前面的行数记进 meta。
- 缺失写作 na，必须读成缺失。
- 开发内划分 42,000 / 9,000 / 9,000，种子 7；行号要落盘可复原。
- 划分是否分层在文档中未写明，这里默认分层并在返回结果里标出，供协议确认。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

__all__ = ["load_aps_csv", "split_features", "missing_profile", "dev_split",
           "file_sha256", "binarize_labels"]

NA_TOKENS = ["na", "NA", "Na", "NaN", "nan", "?", ""]

# APS 的 class 列写作 pos / neg（pos = APS 相关故障，neg = 其他故障），不是 0/1
LABEL_MAP = {"pos": 1, "positive": 1, "1": 1, 1: 1,
             "neg": 0, "negative": 0, "0": 0, 0: 0}


def binarize_labels(series):
    """把标签列统一成 0/1。遇到没见过的取值直接报错，不静默当成负类。"""
    s = pd.Series(series)
    if s.isna().any():
        raise ValueError("标签列存在缺失，先确认是否应为排除行")
    uniq = list(pd.unique(s))
    unknown = [v for v in uniq if (v.lower() if isinstance(v, str) else v) not in LABEL_MAP]
    if unknown:
        raise ValueError(f"标签出现未定义的取值 {unknown[:5]}，现有 {uniq[:5]}")
    return s.map(lambda v: LABEL_MAP[v.lower() if isinstance(v, str) else v]).to_numpy().astype(int)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_header_line(path, label_col="class", max_scan=500):
    """返回表头所在行号（0 起）。找不到就报错，不猜。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_scan:
                break
            fields = [f.strip().strip('"').strip("'").lower() for f in line.rstrip("\n").split(",")]
            if label_col.lower() in fields:
                return i
    raise ValueError(f"前 {max_scan} 行内没有找到含 {label_col!r} 的表头，拒绝按默认表头读取")


def load_aps_csv(path, label_col="class"):
    """读入 APS 风格 CSV，返回 (df, meta)。"""
    header_idx = _find_header_line(path, label_col=label_col)
    df = pd.read_csv(path, skiprows=header_idx, na_values=NA_TOKENS, keep_default_na=True,
                     low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if label_col not in df.columns:
        raise ValueError(f"读入后没有 {label_col!r} 列，实际列名前几个：{list(df.columns[:5])}")
    dropped = []
    for col in list(df.columns):
        if col.lower().startswith("unnamed:"):
            df = df.drop(columns=[col])
            dropped.append(col)
    meta = {
        "path": str(path),
        "sha256": file_sha256(path),
        "header_line_index": int(header_idx),
        "n_preamble_lines": int(header_idx),
        "n_rows": int(len(df)),
        "n_cols_including_label": int(df.shape[1]),
        "label_col": label_col,
        "dropped_index_columns": dropped,
        "na_tokens": NA_TOKENS,
    }
    return df, meta


def split_features(df, label_col="class"):
    """拆出特征矩阵与 0/1 标签（pos→1、neg→0）。未知标签取值直接报错。"""
    if label_col not in df.columns:
        raise ValueError(f"缺少标签列 {label_col!r}")
    y = binarize_labels(df[label_col])
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    return X, y


def missing_profile(df, label_col="class"):
    """缺失画像：总体、正类、负类缺失率，以及逐列缺失率。"""
    if label_col not in df.columns:
        raise ValueError(f"缺少标签列 {label_col!r}")
    X = df.drop(columns=[label_col])
    miss = X.isna().to_numpy()
    y = binarize_labels(df[label_col])
    per_col = miss.mean(axis=0) if len(miss) else np.array([])
    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "overall_missing_rate": float(miss.mean()) if miss.size else float("nan"),
        "positive_missing_rate": float(miss[y == 1].mean()) if (y == 1).any() else float("nan"),
        "negative_missing_rate": float(miss[y == 0].mean()) if (y == 0).any() else float("nan"),
        "per_column_missing_rate": per_col.astype(float).tolist(),
        "n_all_missing_columns": int((per_col == 1.0).sum()) if per_col.size else 0,
        "n_no_missing_columns": int((per_col == 0.0).sum()) if per_col.size else 0,
    }


def dev_split(y, sizes=(42000, 9000, 9000), seed=7, stratify=True):
    """按 sizes 切 train/val/calib，返回索引与正类数；不改变行序。

    stratify=True 时按类别比例分配（各类内部打乱后按累计比例切），
    文档未规定是否分层，调用方需把这一选择写进协议。
    """
    y = np.asarray(y).ravel().astype(int)
    names = ("train", "val", "calib")
    if len(sizes) != 3:
        raise ValueError("sizes 必须是 (train, val, calib) 三个数")
    sizes = tuple(int(s) for s in sizes)
    if any(s < 0 for s in sizes):
        raise ValueError("sizes 不能为负")
    if sum(sizes) > len(y):
        raise ValueError(f"划分总量 {sum(sizes)} 超过可用行数 {len(y)}")
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in names}
    if stratify:
        bounds = np.cumsum([0] + list(sizes))
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls).copy()
            rng.shuffle(cls_idx)
            prev = 0
            for k, s in zip(names, sizes):
                # 按累计比例定边界，保证每类分配数之和等于该类行数
                end = int(round(bounds[names.index(k) + 1] * len(cls_idx) / sum(sizes)))
                end = min(max(end, prev), len(cls_idx))
                parts[k].append(cls_idx[prev:end])
                prev = end
            if prev != len(cls_idx):
                parts[names[-1]].append(cls_idx[prev:])
    else:
        perm = rng.permutation(len(y))
        start = 0
        for k, s in zip(names, sizes):
            parts[k].append(perm[start:start + s])
            start += s
    out = {}
    for k in names:
        arr = np.sort(np.concatenate(parts[k])) if parts[k] else np.array([], dtype=int)
        out[k] = arr
    out["sizes"] = {k: int(len(out[k])) for k in names}
    out["positives"] = {k: int(y[out[k]].sum()) for k in names}
    out["seed"] = int(seed)
    out["stratified"] = bool(stratify)
    out["n_total"] = int(len(y))
    return out
