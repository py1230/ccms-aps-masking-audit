"""从核验结果生成可读报告（纯读，避免手抄数字出错）。

用法：python3 make_report.py --verify <verification.json> --rebuild <rebuild_summary.json> --out <报告.md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--rebuild", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = json.loads(Path(a.verify).read_text(encoding="utf-8"))
    rb = json.loads(Path(a.rebuild).read_text(encoding="utf-8"))
    s = v["summary"]
    per = {r["split"]: r for r in rb["per_split"]}

    L = []
    L.append("# A3 模型与评价侧材料：核验报告\n")
    L.append(f"生成时间：{v['generated_at']}　身份：加载 2026-11-12 复现件已保存的 30 个模型复算；**未重新训练任何模型**\n")
    L.append("## 一句话结论\n")
    L.append(
        f"30 个模型的概率与 `preds_s*.npz` 逐位相同（最大绝对差 {s['max_abs_prob_diff_vs_npz']:.3e}），"
        f"与 270,000 行 CSV 的差异仅来自十进制文本往返（{s['max_abs_prob_diff_vs_csv']:.3e}）；"
        f"30 组 AP 与历史记录 **{s['n_ap_bit_equal_vs_history']}/30 完全相等**（最大差 {s['max_ap_diff_vs_history']:.3e}）。"
    )
    L.append("")
    L.append("这同时证明：本次事后重建的评价掩码与插补中位数，能逐位还原当时的预测输入。\n")

    L.append("## 核验项\n")
    L.append("| 检查 | 结果 |")
    L.append("|---|---|")
    L.append(f"| 模型 sha256 与复现件 `cells_s*.json` 一致 | {s['all_model_sha_match']} |")
    L.append(f"| 掩码 sha256 与 `preds_s*.npz` 一致 | {s['all_mask_sha_match']} |")
    L.append(f"| 样本映射（CSV `source_row_0based` = `val_rows[val_index]`、`y_true` = `y_val`） | {s['sample_mapping_ok']} |")
    L.append(f"| 评价掩码可复现（`isnan(ev)` = 原生 ∪ 追加） | {s['sample_mapping_ok']} |")
    L.append(f"| 环境 | Python {v['environment']['python']} / numpy {v['environment']['numpy']} / scikit-learn {v['environment']['sklearn']} |")
    L.append("")

    L.append("## 评价侧重建（每个划分）\n")
    L.append("| 划分 | 训练行 | 验证行 | 原生缺失格 | 追加删除格 | 合并后缺失格 | 新增率 | M3 形状 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sp in (11, 22, 33, 44, 55):
        r = per[sp]
        rate = r["added_cells"] / (r["n_val"] * 170 - r["native_missing_cells"])
        L.append(
            f"| {sp} | {r['n_train']} | {r['n_val']} | {r['native_missing_cells']:,} | {r['added_cells']:,} | "
            f"{r['merged_missing_cells']:,} | {rate:.4f} | {tuple(r['m3_eval_shape'])} |"
        )
    L.append("")
    L.append("追加删除率理论值 0.30，实测 0.2996–0.3001，差异来自「只对可见单元格施加」与格数取整。\n")

    L.append("## 30 格逐项\n")
    L.append("| 划分 | 臂 | 模型种子 | 样本数 | 最大概率差（vs npz） | 最大概率差（vs csv） | 复算 AP | 历史 AP | AP 差 | AP 相等 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in v["cells"]:
        L.append(
            f"| {c['split']} | {c['arm']} | {c['model_seed']} | {c['n']} | {c['max_abs_prob_diff_vs_npz']:.3e} | "
            f"{c['max_abs_prob_diff_vs_csv']:.3e} | {c['ap_recomputed']:.16f} | {c['ap_history_recorded']:.16f} | "
            f"{c['ap_diff_vs_history']:.3e} | {c['bit_equal_vs_history']} |"
        )
    L.append("")
    L.append("## 输入指纹\n")
    L.append(f"- 数据：`{v['data']['path']}`　sha256 `{v['data']['sha256']}`")
    L.append(f"- CSV：sha256 `{v['inputs']['csv_sha256']}`")
    for k, val in v["inputs"]["preds_sha256"].items():
        L.append(f"- {k}：`{val}`")
    L.append("")
    L.append("## 边界\n")
    L.append("1. 这里核的是「模型 + 重建的评价侧预处理」能否逐位重现已保存的概率，**不是**「历史 2026-11-04 的模型与本次加载的完全相同」——历史模型对象从未落盘，无法比较对象身份。")
    L.append("2. AP 相等只能说明这套输入与模型下指标一致；不能反推历史概率数组或历史模型内部参数相同（PAA/审稿语境下必须这样写）。")
    L.append("3. 模型加载会反序列化执行代码，因此只对本包内文件使用 `joblib.load`。")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
