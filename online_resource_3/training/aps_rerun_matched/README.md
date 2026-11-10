# 历史版本环境全量复跑

本目录包含独立的125模型全量运行。详见PROTOCOL.md补充与上级交付总览。

使用解释器：`/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。Python3.12.15、NumPy2.5.4、scikit-learn1.9.1；完整依赖见environment_packages.txt。

顺序执行：run_aps.py --check；run_aps.py；verify_results.py；compare_historical.py。已有输出拒绝覆盖，重复实验用make_repeat.py建立新目录。

与上级第一次运行保持相同科学配置，仅联合软件环境不同。报告对照以新计算结果为准。
