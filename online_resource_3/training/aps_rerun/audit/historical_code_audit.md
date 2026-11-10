# APS 历史实现与输入审计

2026-11-08。由同一会话上下文的子助手执行；只读历史材料，未训练模型。本文不构成历史来源认证。

## 来源入口

本地完整包根目录（下文 PKG）：

`/Users/bgc/Documents/ChatGPT/research/rr/02_原完整复现包/CCMS论文修订完整资料包_2026-11-01_192149`

历史项目根目录（下文 ARCHIVE）：`PKG/03_原始项目归档/CCMS核查_2026-10-31`。

投稿包（下文 ZIP）：`/Users/bgc/Desktop/CCMS_复现修正_2026-11-05/投稿包_修订/04_Online_Resources/ESM_2.zip`。

| 用途 | 精确来源 |
|---|---|
| 主验证训练入口与旧选参器 | `ARCHIVE/scripts/run_aps_formal.py` |
| 官方测试训练入口 | `ARCHIVE/scripts/run_aps_testset.py` |
| CSV 读取与自定义划分 | `ARCHIVE/src/aps_io.py` |
| 预处理与 HGB 工厂 | `ARCHIVE/src/pipeline.py` |
| C/B 掩码算子 | `ARCHIVE/src/ccms.py` |
| 13 个评价输入 | `ARCHIVE/scripts/run_cb_firstlook.py::build_eval_inputs` |
| α=.25 重建预测入口 | `PKG/02_本轮代码与结果/scripts/run_supplement.py::aps` |
| 主结果保存预测及索引 | `PKG/02_本轮代码与结果/results/aps_seed{11,22,33,44,55}.npz` |
| 完整 α 网格的历史验证指标 | `ARCHIVE/records/aps_formal_20261031T142351.json` |
| 历史官方测试指标 | `ARCHIVE/records/aps_testset_20261031T145423.json`，各划分只覆盖实际运行配置，不是全候选完整覆盖 |
| 完整候选选择规则重放 | `PKG/02_本轮代码与结果/results/archive_audit.json` 的 corrected_selection |
| 历史重建指标、删除计数及环境 | `PKG/02_本轮代码与结果/results/aps_reconstruction.json` |
| A3/A4 训练实现 | ZIP entry `ESM_2/08_repro_bundle/revision/analysis/run_arm3_countmatched.py` |
| A3/A4 保存预测 | ZIP entry `ESM_2/08_repro_bundle/revision/analysis/arm3_seed{11,22,33,44,55}.npz`；同时含 A3、A4，另有 A4 单独导出文件 |
| 旧 A3 保真自述 | ZIP entry `ESM_2/08_repro_bundle/revision/analysis/r9_arm3_fidelity.json` |

## 核实后的历史流程

1. **读取数据。** 扫描含 `class` 字段的表头；`na` 等缺失标记转 NaN；标签 pos=1、neg=0；按 CSV 原列序保留全部 170 个特征。
2. **划分。** 历史 `dev_split` 是自定义函数，不是 sklearn 的 `train_test_split`。使用 `default_rng(seed)`；依 `np.unique(y)` 顺序逐类 shuffle，以累计比例和 round 切分；合并各类索引后，各部分按原始行号升序排列。
3. **预处理。** `MedianImputer().fit(X_train)` 只执行一次，使用未追加遮蔽的当前 42,000 行训练矩阵。全空列中位数置零，不删列。所有实验臂使用同一填补器。缺失指示变量从各臂追加遮蔽后的矩阵重新生成，再与填补后的值拼接，M3 为 340 列。
4. **模型。** `_make_model(2026)` 仅执行 `HistGradientBoostingClassifier(random_state=2026)`，其他参数取运行环境默认。特别是 `early_stopping='auto'`；42,000 行时启用内部验证早停。外部 9,000 行验证用于 α 选择。历史流程没有在 α 选择后用训练+外部验证重新拟合。不能把新 SECOM 的关闭早停与重新拟合规则套到 APS。
5. **C/B 随机数。** 两个算子均按 `default_rng(mask_seed).random(X.shape)` 生成完整矩阵随机数；同一 seed、同一矩阵形状下共享基础随机流。C 概率按行类别决定；B 概率按列决定，预算由训练标签统计标定。两者只删除当前可见单元。
6. **A3/A4。** A3 根据同一 mask_seed 的 C 实际逐列删除数，按列升序对原始可见单元均匀无放回抽样，RNG seed=`mask_seed*1000+7`。A4 同理匹配 B，seed=`mask_seed*1000+11`。代码包含预算超过可见数时截断分支，但随后逐列计数断言必须相等；新实现宜直接报错。不能继承旧脚本文首“pure within-class allocation effect”的因果措辞。
7. **评价压力。** S0 原始输入；S1 10% 与 30% 均使用 101、202、303，以完整形状随机数与可见性相交；S2 seeds 1101、1202、1303，先无放回抽取 round(0.5*n) 行，再按 0.6 追加删除；S3 seeds 2101、2202、2303，无放回抽取 max(1,round(0.3*p)) 列，使整列不可用。评价输入不使用标签。
8. **聚合。** 每个输入先平均三个训练掩码模型的概率形成 ens3，再算 AP；压力族内对三个输入 AP 求平均。不等于先求三个模型的 AP 再平均。
9. **选择。** 历史正式脚本只让非零候选竞争，全部非零不可行才回退 α=0；新任务应采用已修订的完整候选竞争。可行性为 S0 AP 相对基线损失≤0.01；目标为 S0 与三个 S1 10% 输入中最差 AP 最大，并列选更小 α。该修正须明确记录，不隐写为历史原代码行为。

## 本次实际核验

`historical_index_check.json` 记录了五个划分：

- 从当前原始训练 CSV、现存历史 `dev_split` 生成的训练/验证索引，与本地 `aps_seed*.npz` 逐元素一致。
- 五份本地 NPZ 与 ZIP 内同名 NPZ 的 SHA-256 一致。
- ZIP 中 A3 NPZ 的训练/验证索引与对应主分析 NPZ 一致。
- 每次划分为 42,000/9,000/9,000，正类 700/150/150；训练、验证、预留共同覆盖 60,000 个原始行号。

`historical_mask_count_check.json` 记录了 5 个划分×3 个掩码种子×C/B 两臂，共 30 组 α=.25 掩码重建。逐列与逐类×列删除计数 30/30 与 `aps_reconstruction.json` 一致。这是计数一致性，不是逐单元掩码一致性；历史文件没有完整掩码可供逐单元比较。

本地历史源码与 ZIP 对应源码逐字节一致，SHA-256 如下：

| 文件 | SHA-256 |
|---|---|
| `src/aps_io.py` | `787558c6badffb88d3badefc87b454bafd443bdf411714eeecabc325e8bb3f6f` |
| `src/ccms.py` | `2aa211e6ac688c7b1687a1946c30b780a94a3d856a9d0cb59504c0e2554511c2` |
| `src/pipeline.py` | `45cb6f629e0126f3afd8b1bec4b4b4fb290d0a6888c0c1957cd2202cc0a3a26a` |
| `scripts/run_cb_firstlook.py` | `682912992b664996431c9b54c545002e793694680a99fcbf2f12a44c87152809` |
| `scripts/run_aps_formal.py` | `a2f9eee5046d8e806cfbd7a24db0b6038bff2c728ef6c39553970ab1937325c0` |
| `scripts/run_aps_testset.py` | `ccd134addedbde6864eefcf3bcb475f86c52efe2150bba928f9c6a4716724793` |

## 来源缺口及差异诊断要求

- `aps_reconstruction.json.environment.script_sha256` 记载 `90b0fc35f4b122809b6ebb90ce7f461120a45eeef451518d019d8aaf331439d7`，与现存 `run_supplement.py` 的 `e68bb0f7a05a55d129c66048414b6c4b4aed8d52f98a4a2b5ebfa27fa9fb104a` 不同。此缺口覆盖 APS，不仅是 SECOM。可能涉及路径或其他版本变更，但没有匹配原件时不能判定原因。该 JSON 的 pipeline 哈希则与现存文件一致。
- 旧 JSON 自述 Python 3.12.15、NumPy 2.5.4、sklearn 1.9.1；本次应实测并保存当前版本，不把旧元数据视为当前环境。跨版本默认参数与底层实现差异应列为解释候选，不能未经同环境控制就宣称已解释全部 AP 差异。
- 旧 `aps_reconstruction.json` 记载其五次重建对旧验证 AP 最大差约 2e-16–4e-16；A3 fidelity JSON 自述 M3/C025 预测零差。以上是归档记录，本次并未执行这些模型训练。
- v2.4 附录提及后来“独立复现尝试未能复现 C 臂”。在本次已读 ESM_2 与指定 rr 范围内，未定位到对应失败尝试的完整脚本、环境和预测原件。因此应保留该来源不确定性，不能把它与旧 A3 零差记录当作同一次运行。
- 新结果与历史结果比较时，应优先按保存预测复算旧 AP；只有候选汇总分数的单元标明“历史逐样本预测缺失”。对应新掩码与计数、索引、预处理检查通过，也只能逐项排除差异来源，不足以证明模型来源。

建议后续首先锁定上述实现，执行新候选全量训练与模型保存。若发现 C 的 AP 差异，按索引、预处理、中位数、缺失标记、随机掩码、HGB 参数与实际迭代次数、概率聚合顺序逐项诊断；未证实的环境解释保留为假设。
