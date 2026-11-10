# APS 训练样本分层重抽与完整选择程序

先读 `PROTOCOL.md`。正式任务为固定split11下20次分层有放回重抽，每次19模型，加一个独立新anchor的19模型，共399次正式拟合。接口资源smoke另1模型，不计入399。完成须全部21unit的summary complete且独立核验通过；中间文件不是完成证明。

## 环境与材料

匹配Python：`/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。固定Python3.12.15、NumPy2.5.4、sklearn1.9.1；其它安装版本见environment_packages.txt。关闭早停、固定100轮，所有模型重新训练，不能将旧早停模型作为本轮anchor。

- `raw_data/`：训练/测试原始CSV。完整有序X由对应CSV与train_rows重建，重复ID保留多次。
- `prepared_indices/`：anchor及90101–90120重抽的完整有序ID、60,000行multiplicity、原始训练池/验证/预留ID、seed/RNG状态。test行号属于另一个CSV命名空间。
- `validation_masks.npz`、`test_masks.npz`：固定4验证及13测试压力输入。
- `units/{anchor,seed90101,…,seed90120}/`：各19模型、预处理、完整训练mask、训练/评价矩阵指纹、单模型和候选集成概率、候选AP、选参和时间日志。
- `freeze.json`：正式拟合前科学代码、协议、配置、原数据及计划索引指纹；后处理和核验文件由交付清单另行覆盖。
- `interface_smoke/`：2,000行接口/资源试验，仅验证预测完整性，不算AP；smoke时源码快照与记录SHA完全匹配。
- `resample_identity_summary.json/csv`：各类unique原ID、重复数及最大multiplicity等，无性能诊断。
- `descriptive_summary.json/csv`、`ap_by_unit.csv`、`contrasts_by_unit.csv`、`selections.csv`：20次重抽均值、样本SD、最小/最大及单列anchor。范围不是置信区间，不作显著性或等效性判断。

## 独立准备与重复

使用匹配解释器执行 `python make_repeat.py ../aps_resample_repeat_01`，进入新目录后：

```sh
python verify_results.py --preflight
python run_resample.py --freeze
python run_resample.py --check
python launch_runs.py
python summarize_results.py
python verify_results.py --unit anchor
python verify_results.py --unit seed90101
```

以相同方式逐个核验seed90102至seed90120，全部21unit独立核验通过后运行 `python verify_results.py --final`。也可逐unit顺序训练，例如 `python run_resample.py --unit anchor`。launcher最多3进程、每进程2线程，7200秒/unit硬上限；失败保留并暂停后续调度，不自动重试。已有输出拒绝覆盖。

这里的python须替换为匹配环境解释器。make_repeat仅复制数据、计划索引及代码，不复制已拟合模型、预测、旧冻结或核验结果，不会自动训练。新目录再次冻结记录实际运行环境；input_sources中的旧绝对路径仅为来源定位，重复执行依赖本包相对路径。代码/科学方案变更须另建版本，不能通过迁移工具冒充同一冻结实验。

20个重抽数据集来源于同一个固定经验训练池；重复ID的副本不是新的独立观测。固定验证/测试样本、类别数量、外层划分及随机种子限制了范围的含义。档案一致性与新训练不能证明历史来源，也不能替代外部现场数据。
