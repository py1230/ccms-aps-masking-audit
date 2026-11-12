# CCMS v2.2 复核脚本与结果包

日期：2026-11-08

本包包含基于 ESM_2.zip 已保存预测与归档候选分数生成的复核脚本和结果。所有分析均为预测/汇总层面的复核，不代表从原始数据重新训练模型。

## 文件

1. `secom_bootstrap_recompute_20261108.py` / `secom_bootstrap_20k_20261108.json`
   - 对 SECOM 五折已保存预测执行 20,000 次折内、类别分层、配对条件 Bootstrap。
   - 固定模型和已保存预测；不重训、不重新选择 alpha。
   - 折内按正负类有放回抽样并保持各折类别数；同一抽样权重用于 C/B 及所有 S1_30 评价掩码重复。
   - 报告五折平均 AP 差及 95% 百分位区间。
   - 不覆盖外层训练集重叠造成的模型拟合依赖。

2. `replay_selector_from_archived_scores.py` / `selector_replay_20261108.json`
   - 从 ESM_2 的 `archive_audit.json` 读取已保存候选验证 AP 表。
   - 独立重放可行性约束、最差四输入 AP 目标和并列规则。
   - 仅验证选择逻辑条件于归档候选分数；不重新计算 alpha=0.5/1.0 的逐样本 AP。

3. `run_r15_decision_replay.py` / `r15_decision_metrics_recomputed_20261108.json`
   - 仅修正 ESM_2 原脚本的旧目录路径和输出目录。
   - 其余计算逻辑保持不变，使用 1,000 次配对 Bootstrap。
   - 复算的 `results` 与 `paired_recall_differences` 与 ESM_2 归档 r15 JSON 完全一致。

## 重要边界

- APS 的 20,000 次 C/B/M3 区间与多重校正另见修订后的 Online Resource 3。
- 本包不包含 APS/SECOM 原始特征数据，也不执行模型重训练。
- SECOM 时间顺序实验尚未运行；对应原始 UCI 文件须先满足预设 SHA-256 门禁。
