# 历史 C 复现表述来源与 APS 数值交叉核对

日期：2026-11-08。此次补查只读历史文件，不训练模型；未修改冻结的 runner、config 或 runner_code_review。

## “独立复现尝试未能复现 C 臂”的来源定位

本次定向检索范围：

1. `/Users/bgc/Desktop/CCMS_复现修正_2026-11-05/投稿包_修订/04_Online_Resources/ESM_2.zip` 中的文本、脚本及 JSON 条目。
2. `/Users/bgc/Documents/ChatGPT/research/rr/02_原完整复现包/CCMS论文修订完整资料包_2026-11-01_192149`。
3. `/Users/bgc/Documents/ChatGPT/论文2.0`，文本检索排除本次新 APS 运行目录，避免循环引用本次报告；另直接抽取根目录 v2.2/v2.3 DOCX 的相关段落。

扫描 267 个本地文本/脚本/JSON 文件、380 个 ZIP 文本条目，解析其中 386 个 JSON。搜索中文“未能复现/复现失败/独立复现”与英文 reproduction/fail/fidelity 等相关措辞，并检查可疑失败状态字段。机器记录见 `c_failure_source_search.json`。

结论：**在上述范围内，未定位到与该句对应的具体失败运行脚本、日志、环境快照或预测原件。** 该句已经出现在当前目录 v2.2 DOCX 的历史 α 网格段落中，v2.3 与 v2.4 沿用，不是 v2.4 新增的实测结果。当前 v2.2/v2.3 修改清单没有提供该失败尝试的明确产物路径。

检索命中的失败字段来自两类无关内容：APS 数据文件名中的 `failure`，以及 `mutation_check.json` 故意植入错误后被测试捕获的 `failed_tests`；均不是对应的 C 训练复现失败记录。

ESM_2 中 `08_repro_bundle/REPRO_STATUS.md` 明确说其打包核验没有重新训练模型。`revision/analysis/r9_arm3_fidelity.json` 与 `notes_c.md` 则自述另一历史阶段 M3/C025 的预测零差。不能以这些记录代替未定位的失败尝试，也不能因为未找到原件就推断该尝试是否发生、历史训练是否真实或失败原因是什么。

## 历史保存预测与两套 JSON 的实际交叉核对

读取完整包 `02_本轮代码与结果/results/aps_seed{11,22,33,44,55}.npz`，用本次当前环境的 `sklearn.metrics.average_precision_score` 独立计算全部保存的 M3/C025/B025 AP，共：

`5 划分 × 3 配置 × 2 评价集 × 13 输入 = 390 单元`。

| 参照 | 比较范围 | 最大绝对 AP 差 |
|---|---|---:|
| `02_本轮代码与结果/results/aps_reconstruction.json` | 全部 390 单元 | 0 |
| `ARCHIVE/records/aps_formal_20261031T142351.json` 与 `aps_testset_20261031T145423.json` | 前者对应验证、后者对应官方测试，全部 390 单元 | 4.440892098500626e-16 |

ARCHIVE 指完整包中的 `03_原始项目归档/CCMS核查_2026-10-31`。逐项 AP、差值和来源类别见 `historical_ap_crosscheck.json`。

因此，在三种有保存逐样本预测的配置上，没有发现“formal/官方 test JSON 与保存 NPZ 使用不同数值”的迹象。此检查没有补齐 α=.5/1 的缺失逐样本预测，也没有验证任何历史模型训练过程。

## 环境与执行参数仍存在的证据限制

- `aps_reconstruction.json.environment` 自述 Python 3.12.15、NumPy 2.5.4、sklearn 1.9.1、macOS、threads=2。它没有保存完整 HGB 参数展开值、每个模型的实际迭代数或早停曲线。
- 旧 formal JSON 的 `fit_info` 主要是拟合时间、样本数和特征数；旧官方 test JSON 也不含实际 `n_iter_` 或 early-stopping 状态。现存 `pipeline._make_model` 仅指定 random_state=2026，故只能从代码确认“使用运行版本默认值”，不能从上述记录恢复全部当时有效默认参数与具体早停轨迹。
- APS reconstruction 的生成脚本期望哈希为 `90b0fc35f4b122809b6ebb90ce7f461120a45eeef451518d019d8aaf331439d7`，现存 `run_supplement.py` 为 `e68bb0f7a05a55d129c66048414b6c4b4aed8d52f98a4a2b5ebfa27fa9fb104a`。包内路径说明记载了脚本副本路径替换，可作为版本变化的线索，但本次没有恢复匹配的原始字节文件，因此不能据此断言差异仅来自路径修改。
- 该 JSON 的 `source_pipeline_sha256=45cb6f629e0126f3afd8b1bec4b4b4fb290d0a6888c0c1957cd2202cc0a3a26a` 与现存 pipeline 源码一致。

新重跑若出现 AP 差异，应继续报告其量值与已核验的输入、索引、计数、预处理和参数事实。软件版本差异可以列为待检验解释，不应自动认定为全部差异的原因。
