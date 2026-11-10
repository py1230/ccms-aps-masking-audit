# APS 主分析重跑材料

本目录保存 2026-11-08 两次全量重跑、一个四模型环境诊断、代码与历史参照。最终结论以 `新旧差异与论文更新建议.md` 为准。`historical/` 是从 ESM_2 原包复制的历史参照，不是本次训练结果。

根目录 `run_results/` 使用sklearn 1.7.2，作为首次环境运行保留；`environment_matched_run/run_results/` 使用历史记录匹配的sklearn 1.9.1环境，是检查历史数值能否恢复的完整新运行。各目录分别有 `verification.json`。阅读复现结论时优先查看最终报告和 `environment_matched_run/`，不要混用两次结果。

## 阅读顺序

1. `新旧差异与论文更新建议.md`：新结果、差异和可支持结论。
2. `PROTOCOL.md`、`config.json`、`freeze.json`：实际设计、显式参数、运行前文件固定。
3. `verification.json`：独立代码的保存产物核验；不是另一轮独立重新训练。
4. `new_vs_historical.csv` 与其他 comparison 文件：逐项新旧对照、缺失历史参照和选择差异。
5. `audit/`：历史索引、计数、代码审查与来源限制。

本实验已接触过历史结果，官方测试是回顾性评价集；结果不构成预注册确认性试验。保存文件一致与模型重载一致不能证明历史训练来源。

## 首次环境运行命令

实际解释器 `/opt/anaconda3/bin/python`，Python 3.13.9、NumPy 2.3.5、pandas 2.3.3、scikit-learn 1.7.2、SciPy 1.16.3、joblib 1.5.2；完整版本在 `environment_packages.txt`。两线程；不需要下载数据或升级现有环境。

在本目录执行的顺序为：

```sh
/opt/anaconda3/bin/python run_aps.py --smoke
/opt/anaconda3/bin/python run_aps.py --check
/opt/anaconda3/bin/python -u run_aps.py > stdout.log 2> stderr.log
/opt/anaconda3/bin/python -u verify_results.py > verification_stdout.log 2> verification_stderr.log
/opt/anaconda3/bin/python compare_historical.py
```

`--smoke` 是小规模接口检查；正式运行前固定科学配置，随后 `--check` 核验冻结输入。现有结果目录已存在时，训练会拒绝覆盖；核验报告也拒绝覆盖。

## 如何另开一次重复运行

使用 `make_repeat.py` 创建不存在的新目录，复制冻结代码与历史参照，核验原始数据哈希，并仅更新绝对路径和新冻结时间，不复制旧结果、不启动训练。例如在本目录执行：

```sh
/opt/anaconda3/bin/python make_repeat.py ../aps_rerun_repeat_01 --data-dir ../../data/raw/aps
```

然后进入新目录执行 `run_aps.py --check`、`run_aps.py`、`verify_results.py`、`compare_historical.py`。原始数据所在项目路径是 `data/raw/aps/`。请使用上述相同依赖版本；跨版本的数值变化必须作为新的环境差异记录。该工具复制历史参照，所需空间以当前目录文件大小为准。

完整 ZIP 另外包含 `raw_data/` 下的两份原始 CSV。独立解压后可在包目录运行 `make_repeat.py ../aps_repeat_01 --data-dir raw_data` 创建新运行。`manifest.json` 覆盖包内有效载荷（不含清单自身），`package_verification.json` 在本地记录归档校验结果与 ZIP 指纹。

要重复历史版本环境运行，应从 `environment_matched_run/` 内调用其 `make_repeat.py`，使用该目录记录的Python3.12.15、NumPy2.5.4、scikit-learn1.9.1等版本。独立解压后数据参数为 `--data-dir ../raw_data`；在当前项目中也可直接传入原始APS数据目录的绝对路径。该环境解释器路径见该目录 README。创建目录不会自动训练，实际执行仍须使用相应环境。

## 产物布局

每个 `run_results/split<seed>/` 包含：

- `preprocessing.npz`：未遮蔽训练部分的中位数、列名、缺失率、行号。
- `models/`：25 个单模型及其预处理和训练元数据。
- `training_masks/`：新增删除布尔掩码；M3 为全零。
- `diagnostics/`：逐行、逐列、逐类删除计数及预算；A3/A4 的实际列预算与参考臂精确匹配。
- `evaluation_masks.npz`：验证压力输入；测试掩码由所有划分共用，保存于 `run_results/test_evaluation_masks.npz`。
- `predictions/`：各单模型的验证和测试预测，附行号与标签。
- `ensemble_predictions.npz`：概率平均后的 ens3；`metrics.json`：逐输入 AP。
- `selection.json`：测试预测前固定的完整候选选择；`fit_records.json`：实际早停迭代数和拟合时间。

原始缺失由原始 CSV 恢复；保存掩码仅表示新增删除。原始样本行号从零起算，排除 CSV 说明和表头行。所有压缩 NPZ 都可用 `numpy.load(..., allow_pickle=False)` 读取。joblib 模型只应从可信的本项目文件加载。

统计范围为描述性点估计与选择重放，未给新预测套用旧置信区间。训练掩码种子是集成的组成部分，五次划分相互重叠；均不能作独立数据集计数。
