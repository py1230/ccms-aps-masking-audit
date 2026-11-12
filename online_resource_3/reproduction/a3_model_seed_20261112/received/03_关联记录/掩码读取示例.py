"""从 preds_s*.npz 还原某折某臂的训练掩码位图。

True 表示该训练单元格在本次遮蔽中由观测变为缺失。
"""
import numpy as np

z = np.load("02_复现件_模型与掩码_非原件/A3_repro_2026-11-12/preds_s11.npz")
shape = tuple(int(x) for x in z["delmask_shape"])          # (42000, 170)
bits = np.unpackbits(z["delmask_C"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
print(shape, int(bits.sum()))                               # 新增遮蔽单元数
print("mask_sha256:", z["mask_sha256"])                     # 与对照表逐格核对
# B 臂：把 delmask_C 换成 delmask_B
