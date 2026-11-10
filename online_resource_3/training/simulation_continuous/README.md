# 独立模拟种子扩展与完整 X 归档

本目录是 v2.5 后的新实验，不覆盖或合并历史 501–505 五种子结果。正式种子固定为 601–630，每个种子 20 次拟合，共 600 次；完成状态以 summary.json 的 status 和独立核验报告为准。统计方案见 PROTOCOL.md，参数见 config.json。完整 X、Y、划分、随机数、掩码、全部候选模型及预测均按种子归档。

## 执行与冻结

运行环境解释器为 `/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。下列命令中的 python 均应替换为该解释器，或在保持 environment_packages.txt 版本一致的隔离环境运行。训练核验 Python、NumPy、SciPy、scikit-learn 和 joblib 版本，threadpool 限定 2 线程，最多 3 个并发子进程。

```sh
python run_simulation.py --smoke
python freeze_run.py
python run_simulation.py --check
python -u launch_simulation.py > launch_stdout.log 2> launch_stderr.log
python analyze_results.py
python verify_results.py
```

以上训练命令用于全新运行副本。已存在的 seed 输出目录、正式运行目录或冻结文件会被拒绝覆盖，不能用同一结果目录隐式重试。需重新训练时，新建目录，仅复制代码、source/、historical/、config.json、PROTOCOL.md 与审查资料，在新目录完成接口检查与独立代码审查后重新冻结。不要复制旧 freeze.json、结果或 smoke 回执后当成新运行。

可用 `python prepare_repeat.py ../simulation_repeat_01` 创建没有结果及冻结回执的新副本。工具会核对所复制文件的字节哈希；复制的审查是原始代码审查快照，不能称为新独立科学审查。若科学代码和参数未变，可核对该快照适用性并按上述命令运行新的 smoke 与 freeze；任何科学改动都应另做审查后再运行。本轮固定的 30 种子列表不会由此工具自动扩展；用同一列表重跑是计算复现，不是新增独立样本。

首次接口检查在独立 seed 99001 上完成 20 次拟合。代码审查随后加强了启动失败记录与源码哈希绑定，首次全部产物保留于 smoke_history/attempt1/。第二次在同一接口 seed 99001 运行完整 20 模型；随后冻结发现匹配环境无 pip，环境清单改用 importlib.metadata 枚举已安装包，第二次产物保留于 smoke_history/attempt2/。第三次接口检查绑定该最终冻结脚本。故共 60 次接口拟合，不能把三次相同种子的接口运行视为独立模拟重复；正式 600 次拟合另计。冻结失败记录另存 audit/freeze_attempt1_failure.json。

冻结记录绑定协议、精确配置、训练及分析代码、复制源文件、环境与审查。源码复制与新运行一致性不证明历史训练来源，也不补回历史缺失的 X 原件。

`source/` 的四个模块足以执行新模拟，不依赖历史结果目录。`historical/run_supplement.py` 仅作源审查留档，含旧目录依赖且不会直接执行。`audit/historical_anchor_check.py` 等历史锚点脚本依赖本机既有历史资料路径，不能仅凭本包重新执行那部分历史比较；其输出属于本次历史审计记录。新 30 种子从生成到训练、分析和独立核验可由本包的主体脚本完成。

## 每个种子的文件

- `data.npz`：8000×20 完整有限值 X、y、train/val/test 行索引、b、delta、U_label、U_train/U_val/U_test、三组 U_mask 与三组 U_stress。目标九个 m 共用唯一 U_test。
- `preprocessing.npz`：仅原始训练矩阵拟合的 medians、经验 q、类别 counts、missing_counts。
- `evaluation_masks.npz`：源训练/验证自然缺失、三组验证新增压力掩码、九个目标缺失掩码。新增压力掩码不含自然缺失，需要取二者并集。
- `models/{tag}.joblib`：model、medians、train_rows、方法与种子元数据。仅加载本次可信运行生成的模型。
- `masks/{tag}.npz`：每模型新增训练掩码、期望预算与行列/类别计数；M3/M4 是零新增掩码。C/B 匹配期望逐列预算，不宣称实际删除计数完全相等。
- `predictions/{tag}__val.npz` 与 `__test.npz`：行索引、标签与该单模型所有条件的概率。
- `ensemble_predictions.npz`：M3/M4 单模型、六个非零候选三模型概率均值、C_selected/B_selected 的完整概率与对应标签/索引。
- `selection.json`：C/B 的全部候选及选中值；α=0 候选共用 M3。选参完成写盘后才构建目标缺失矩阵和预测，先后顺序由 events.jsonl 记录。
- `metrics.json`：10 配置的 val/test AP；`result.json` 为该种子的摘要，metrics 字段只存九点目标 AP，与历史 JSON 结构兼容。
- `fit_records.json`、`events.jsonl`、stdout/stderr：全部拟合与运行记录；失败时保留 failure.json，不自动重试。
- `manifest.json`：该种子有效载荷哈希与字节数，不含清单自身及之后写出的 result.json；全包最终清单须覆盖这些外层文件。

非零 tag 格式如 C025_seed9107；集成 tag 为 C025。条件名为 S0、S1_10_101/202/303 及 gap_1、gap_0.75、…、gap_-1。同种子 C/B 共用基础随机数、原始 X/Y、划分和所有评价输入。

## 完整性与解释

`summary.json` 只有全部 30 种子成功时标记 complete。失败种子不按效果筛除；launcher 保留启动失败、进程退出及超时记录，固定 1800 秒单种子硬超时。目标数据不得用于选参，也不会在目标结果出来后追加种子以追求显著性。

主要估计量、18 个次要比较、多重比较及配对向量 bootstrap 由冻结的 analyze_results.py 计算。0.005 AP 半宽是本实验的精度目标，不是应用最小重要差异，也不预先保证达到。模拟区间只覆盖本生成器下独立数据种子的变化，不能推断工厂总体或部署阈值。
