# 最终中文实验报告复核

结论：**通过，可进入归档前完整性检查。** 未发现报告数值错误、评价集混淆或机制解释越界。本记录仅审查下列报告快照，不替代随后 ZIP 校验。

## 数值和证据映射

- 主表 3 渠道×4 条件/比较的 12 个单元、每格均值/最小/最大共 36 个数值，与已独立核对的 descriptive_summary.json 逐项一致；除 MO 选择后 S0 均值以六位小数显示为 −0.000034，其余按四位小数显示。
- 每重复 S1 30% 表的 18 个六位小数差值逐项一致；C/B 选零频数 6 项及方向计数一致。
- M2 固定官方测试 S1 30% 为两负一正；M3、MO 均三组负向。报告没有把 M2 的变号扩张到验证集或把三组范围写成置信区间。
- 全局 verification.json 已通过：855 模型、22230 单模型预测数组、8190 AP、90 项选择及 15 组跨渠道配对检查。重载预测误差 0，AP 最大误差 4.44e-16；报告所列其他最大误差与原 JSON 一致。
- 基线锚点报告为 passed、130 数组、最大差 0，正文明确仅为第一组 M3 基线，不要求新遮蔽候选等于历史结果。

## 解释边界

明确区分固定强度与完整选择；M3/MO 选择 C=0 后等于自身基线，不能把相对 B 的差值视为实施非零 C 的损失。渠道比较不作纯因果解释；固定行、固定划分和固定评价压力的三个联合随机组不能代表总训练不确定性。回顾性测试、历史来源限制、未完成 A5/模拟扩展等均如实保留。

核验分工被说明为同一会话中的独立代码路径，没有声称盲审、跨模型独立研究或期刊同行评审。旧条件 Bootstrap 区间没有移用到本轮。

终审后显示精度更新：MO 选择后 S0 均值原来四位小数为 −0.0000，现据原值 −3.360724504701448e−05 改为 −0.000034，范围保持不变。逆向还原该唯一文本变化后的 SHA256 与上次审查快照一致，其余报告文字不变。

## 审查快照

- `experiments/aps_channels_repeats_20261108/信息渠道与训练重复_结果报告.md`：`96979920fc01b895c7b80de840fadec54dc3697e5d315c3dfeccb71585b64011`
- `experiments/aps_channels_repeats_20261108/descriptive_summary.json`：`af85ea49a3b7afa049dc3a0d7a4372ad0458699c81bf4e974a0177c2738ee0f6`
- `experiments/aps_channels_repeats_20261108/verification.json`：`e8475e151736d411f19218c08ac29ba88e7d3c299bef12c3d3ec4c8c84b4aaeb`
- `experiments/aps_channels_repeats_20261108/audit/baseline_anchor_check.json`：`60695a4107b1a99cb64327b7d94b5c0d9b2aebd242c27c5bb84001b260591859`
- `experiments/aps_channels_repeats_20261108/audit/summary_review.json`：`dca8ef20a53f5026d5a1e423916eb9a90d2d6a4b3819b951582eeaf90fb18666`
