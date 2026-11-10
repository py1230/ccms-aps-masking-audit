# APS A5 双边际匹配实验材料

先读 `PROTOCOL.md`，再查看 `descriptive_summary.json`、中文结果报告及独立核验。此目录是新A5实验，不覆盖v2.6或旧APS材料。已完成75次新A5拟合，独立核验全部325,369,800次提议、110个模型、1,430组概率数组及AP/汇总/结构诊断。概率重载误差0，AP最大误差4.44e-16；详细结果见A5双边际匹配_结果报告.md。

## 环境与结构

本机匹配解释器：`/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。Python3.12.15、NumPy2.5.4、sklearn1.9.1；见environment.json，完整安装包版本见environment_packages.txt。系统C编译器用于掩码交换，源码及编译记录归档。

- `raw_data/`：原始APS训练/测试CSV。
- `reference/`：M3 rep1五划分的35个配对C025/B025/BASE模型、测试预测及预处理/掩码，来源指纹在reference_sources.json。来源路径只是证据定位，重复运行不依赖它们。
- `pilot/`：单掩码rep0的无性能运行；不训练模型，不计入75条正式链。
- `feasibility/split*/reference*/rep*/`：75个正式mask.npz、trace.npz及chain.json。随机数由批次状态、调用顺序和SHA完整核对；trace存接受attempt及两个边编号。
- `legal_switch_certificate.npz`：每参考256个行不相交合法矩形，给出至少2^256个合法掩码下界。
- `freeze.json`：拟合前科学文件/所有输入及掩码指纹。核验与后处理代码可另行完善，其指纹由交付清单覆盖。
- `run_results/split*/rep*/`：75模型、单模型概率、25个三模型集成预测、全部AP及训练矩阵指纹。
- `descriptive_summary.json/csv`：五随机化重复的均值与最小最大值。范围不是置信区间；主比较S1 30%的C025−A5。
- `mask_structure_summary.json/csv`及`structure_diagnostics/`：75条链的类别总删除率、类别×列计数、改变的行列比例与部分列对共删除结构。此为掩码后处理诊断，不是混合证明。
- `repeat_preparation_check.json`：134个隔离复制文件指纹、迁移目录C重编译和CLI检查通过；此检查没有启动训练，不等同于独立完整重跑。

## 隔离重复

先用匹配Python解释器执行 `python make_repeat.py ../aps_a5_repeat_01`，进入新目录，再执行：

```sh
python run_a5.py --compile
python run_a5.py --feasibility
python mask_structure_summary.py
python verify_results.py --masks-only
python run_a5.py --freeze
python run_a5.py --check
python run_a5.py --run --split 11
python run_a5.py --run --split 22
python run_a5.py --run --split 33
python run_a5.py --run --split 44
python run_a5.py --run --split 55
python summarize_results.py
python verify_results.py
```

上述命令中的python须替换为匹配环境解释器。逐split命令可用提供的launcher并发执行；最多3进程、各2线程，每split 7200秒上限。已存在输出拒绝覆盖。make_repeat只复制输入和代码，不复制旧结果/正式掩码/冻结文件，也不自动训练；新环境重新编译可能改变动态库哈希，应记录实际编译器与环境。运行前应核对已归档环境版本。

此材料可独立进行本轮训练与核验；哈希不证明旧研究来源。A5保持行列删除数却不保持全部联合掩码结构，其随机化目标只是从C开始的20E次有限交换程序，不声称均匀或混合充分，不作纯类别因果解释。
