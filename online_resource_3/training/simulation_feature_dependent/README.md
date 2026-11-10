# 第二机制模拟实验归档

先阅读[中文报告](第二模拟机制实验报告.md)及[预定协议](PROTOCOL.md)。这是全新的30种子实验，未修改或合并旧实验。`freeze.json`内文件不可修改；已有结果目录禁止覆盖或隐式续跑。

运行环境在`environment.json`和`environment_packages.txt`。正式运行顺序为：编译/边界检查 → `run_simulation.py --smoke` → `verify_results.py --smoke` → `freeze_run.py` → `launch_simulation.py` → `analyze_results.py` → `verify_results.py`。完整运行后另执行`export_generation_probabilities.py`、`report_results.py`、`plot_results.py`生成派生留存、报告和图。当前已完成，不应再次直接运行启动器。`prepare.py`仅是最初构建记录，后续核验器有修正，**不可作为重建全部当前文件的唯一来源**。

`analysis/delivery_summary.json`提供机器可读入口，`audit/independent_verification.json`给出全部核验范围；SHA256只支持文件完整性，不认证其他历史实验。模型仅来自本轮本地拟合。
