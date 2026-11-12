# CCMS 归档复核摘要（2026-11-08）

## 已核验

- ESM_2.zip：442 个 ZIP 条目（441 个文件 + 顶层目录项）；SHA256SUMS_full.txt 中 441 项全部匹配。
- Online Resource 3 原包：7 项文件校验全部匹配；修订包已修正 README 与 JSON source path 元数据。
- APS 表 4：执行 `10_repro_check_2026-11-05/reproduce_tables.py`，24 个 AP 单元格按四位小数 24/24 一致；最大绝对差 4.7858e-5。
- APS 20k：24 个 draw arrays（3 个对比 × 8 个数据/压力单元格），每个数组形状 20,000 × 5；八个 C-B 单元格前 1,000 次抽样与旧归档逐元素一致；从 20k draws 独立重算的名义区间、Bonferroni 区间、Holm 调整 p 值和 sup-t 临界值/区间与 summary JSON 一致。sup-t 临界值约 2.9516066；两个 S0 Holm 调整 p 值约 0.0621，六个压力单元格约 0.0004。
- Figure 1：论文内嵌 PNG 与 Online Resource 3 的 `01_three_way_decomposition_20k.png` SHA-256 完全一致；图注应写 20,000 次。
- APS alpha 选择：独立脚本对 5 个划分 × C/B 共 10 个选择进行规则重放，全部与归档候选分数表一致。C 均选 alpha=0；B 依次选 0.25、1.0、0.5、0.5、0.5。
- SECOM：五折测试索引互斥并覆盖 1,567 行；保存预测重新计算的固定/选择后点估计与归档 summary 一致。新的 20,000 次折内类别分层条件 Bootstrap 已生成。
- APS A3/A4：归档 1,000 次 draws 重算 C−A3 和 A3−B 区间，与 r9 JSON 一致。
- 决策层：修正 r15 脚本的旧目录路径后运行 1,000 次 Bootstrap，`results` 与 `paired_recall_differences` 与原归档 JSON 完全一致；Table 10 官方测试 S1 30%、5% 预算 C−B 应为 −0.0302。
- 第二轮补充包：SHA256SUMS 中 12 项全部匹配；build 脚本重建主 AP 表精确一致；补充模型和模拟 CSV 的共同数值列一致，但额外元数据列及其他若干 CSV 未由该脚本重建。

## 仍未关闭

- 冻结协议：39 项中 38 项匹配；唯一不匹配为 `protocol_v2_冻结_2026-10-31.md`。清单期望 SHA-256 与包内实际 SHA-256 不同，期望哈希对应的原始文件未在 ESM_2.zip 中找到。
- 官方测试集历史接触范围仍需作者依据原始实验记录最终确认。
- ESM_2.zip 不含原始 APS/SECOM 特征 CSV、全部模型 checkpoint 或完整历史 stdout；不能声称端到端重训练复现。
- alpha=0.5/1.0 候选的逐样本预测未随 ESM_2 保存；alpha 选择仅完成候选分数表上的规则重放。
- SECOM 修正模型生成脚本的 SHA-256 未在 ESM_2 中找到；新 Bootstrap 仅基于已保存预测，未重训模型。
- SECOM 时间顺序包仅含协议和脚本，没有满足哈希门禁的原始数据和模型运行结果。
- A5 双边际匹配、M2 统一信息渠道、missingness-only 完整统一实验、30+ 种子模拟及额外模型家族实验仍未完成。
