# 后处理与归档说明

预定的数值分析脚本 `summarize_results.py` 和独立核验脚本 `verify_results.py` 已在正式拟合前列入 `freeze.json`。`make_figure.py` 是正式拟合启动后、结果统计前新增的展示脚本，完整展示预定五压力族六条件，不参与训练、选参或数值推断；其身份列入最终归档清单。

`supplementary_checks.py` 是结果汇总后增加的过程及未聚合方向检查，完整输出全部五压力族/六条件的15个划分×后续流值的方向计数和范围，以及模型实际迭代数；它不新增拟合、对比或推断。该15值范围不得与主要表中三个五划分平均值的范围混用。

正式模型环境与文档运行时均没有 Matplotlib。首次图形环境探测在文档运行时产生 `ModuleNotFoundError: No module named 'matplotlib'`；随后只读确认本机既有 Anaconda base 提供该库，使用该环境绘制 CSV 图形，未安装软件或修改正式模型环境。图形运行环境单列在 `analysis/figure_environment.json`。训练、AP统计及独立重放/模型验证仍使用冻结的 Python 3.12.15 历史环境。这不是正式实验失败或更换训练环境。

所有生成、训练和核验输出均使用不覆盖的目录。正式拟合未发生失败、种子替换或重复运行。正式运行的 `seconds` 从原始输入载入后开始计时，包含模型循环、序列化、预测和日志；分项拟合/预测时间不包含所有读写开销。掩码生成时间包括约束检查与轨迹/检查点写入。14400 秒预算在每个模型完成后检查，本次实际循环远低于该预算。

图中菱形为三个后续流的均值，竖线为其最小/最大范围，小圆点为各流结果；每个流先汇总三个参考概率，再计算 AP，随后平均相应压力输入与五划分。不是置信区间，不将独立模型或压力掩码当作独立数据样本。

核心分析/核验完成后，可在同一新运行目录依次执行 `python supplementary_checks.py`、`/opt/anaconda3/bin/python make_figure.py`、`python write_audit_report.py`、`python create_manifest.py` 生成补充描述、图形、编辑交付报告和最终清单。除图形展示外，`python` 均指 README 指定的历史环境。编辑交付脚本会写入该项目的 `revision_v3.2_20261109/audit/a5*` 文件。
