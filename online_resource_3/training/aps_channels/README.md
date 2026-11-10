# APS 信息渠道与条件训练随机性实验

本轮已完成三渠道共855次正式拟合（另3个接口测试小模型）。三渠道进程均正常退出；独立核验通过855模型重载、8,190个AP单元、90项选择与15组跨渠道配对检查，模型重载预测最大误差0，AP最大数值误差4.44×10⁻¹⁶。完整结果解释以 `信息渠道与训练重复_结果报告.md` 为准。

本目录为独立新实验，保留前轮资料。科学问题及解释边界见 PROTOCOL.md，精确配置见 config.json。run_results尚未完成时，不把中间输出当最终结果。

## 执行环境

Python: `/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。

先运行 `run_channels.py --smoke`，完成独立代码审查后运行 `freeze_run.py`，之后 `run_channels.py --check`。正式运行三个命令：

```sh
python -u run_channels.py --channel M2 > M2_stdout.log 2> M2_stderr.log
python -u run_channels.py --channel M3 > M3_stdout.log 2> M3_stderr.log
python -u run_channels.py --channel MO > MO_stdout.log 2> MO_stderr.log
```

这里的python须替换为上述环境解释器。三个命令可在最多三进程中并发运行；每进程2线程。已存在的渠道输出目录拒绝覆盖。正式数据/科学代码/配置更改会导致冻结核验失败。任何重复运行应复制代码/协议/索引到全新目录并在检查数据路径和环境后重新冻结，不复制既有结果。

## 数据布局与核验接口

`run_results/{channel}/rep{1,2,3}/split{11,22,33,44,55}/` 各含19模型：BASE、C025/C050/C100/B025/B050/B100的三个mask seeds。BASE是该channel自己的未新增遮蔽模型，不是固定的M3。

- preprocessing.npz：medians,feature_names,train_rows,q,class_counts,missing_counts。
- training_masks/{tag}.npz：new_mask，原始缺失不计为新增删除。
- diagnostics/{tag}.npz：row_deleted,col_deleted,class_col_deleted,expected_col,q_after。
- models/{tag}.joblib：model,medians,train_rows,arm,alpha,mask_seed,model_seed,repeat,channel,config_sha256。
- predictions/{tag}__val.npz 和 __test.npz：row_indices,y及13个条件名的概率向量。
- ensemble_predictions.npz：train_rows,val_rows,test_rows,y_val,y_test及{BASE|C025|...}__{val|test}__{condition}。
- metrics.json：按配置保存{val|test}__{condition}:AP。
- selection.json：两臂chosen与完整candidates，时间戳；必须先于测试预测。
- fit_records.json：模型文件哈希、有效参数、早停迭代和拟合耗时。
- evaluation_masks.npz：13验证条件的新增掩码；测试掩码位于channel根test_evaluation_masks.npz。

channel/summary.json含cells，每cell含repeat,model_seed,split,chosen_alphas与ap（9配置×val/test×5压力组）。summary status=complete且fits_completed=285才表示该渠道完成。三个渠道共855正式模型；接口smoke另含3小模型。

核验器由不同助手另写，不属于冻结训练实现。核验自己的代码指纹与结果需另存；任何模型重载核验都只加载本目录可信产物。摘要数值与文件哈希一致不证明历史来源。本轮重复只覆盖固定外层数据和评价输入下的联合训练随机性。

## 独立解压与另开重复运行

完整 ZIP 内含 raw_data/ 两份原始 CSV。独立解压后，在解压目录用匹配环境执行：

```sh
python make_repeat.py ../aps_channels_repeat_01 --data-dir raw_data
cd ../aps_channels_repeat_01
python run_channels.py --smoke
python freeze_run.py
python run_channels.py --check
python -u launch_channels.py
python verify_results.py
```

本机直接从当前实验目录创建新目录时，可将数据参数改为 `--data-dir ../../data/raw/aps`。make_repeat只复制代码、协议、索引和源审查记录，核对原始数据哈希并更改CSV绝对路径，不复制结果/冻结/旧smoke，也不会自动训练。源代码审查快照仍指向原始版本；repeat_preparation.json说明仅路径迁移。

使用前应核对environment_packages.txt的完整版本；训练脚本强制核验Python/NumPy/sklearn三项，其余依赖仍应按原环境保持。源码及协议有任何科学改变，须另建版本并重新审查，不能把路径迁移工具当作科学变更审批。

## 汇总报告与归档

最终结果说明为 `信息渠道与训练重复_结果报告.md`。`ap_by_unit.csv`保留每重复/划分/渠道/配置结果；`contrasts_by_unit.csv`和`contrasts_by_repeat.csv`分开给出配对差值和每重复五划分均值；`selections.csv`保留选参，`paired_channel_contrasts.csv`给出配对渠道差值。`descriptive_summary.json/csv`中的范围只是三个重复均值的最小—最大，不是置信区间。报告及全部核验通过后再创建 `APS信息渠道与训练重复完整材料_20261108.zip`。

`manifest.json`覆盖ZIP中的全部有效载荷，原始CSV以raw_data/路径纳入；清单自身和外部`package_verification.json`不纳入自身哈希。打包器逐个流式解压ZIP成员，核对字节数/SHA256及完整成员集合；压缩包指纹保存在package_verification.json。此为交付完整性核验，不能替代科学核验或历史来源证明。

结果汇总脚本为 `scripts/summarize_channels.py`。在三渠道训练完成且独立核验通过后，使用同一Python环境执行 `python scripts/summarize_channels.py`。脚本自动定位本实验目录的config.json；输出描述性CSV/JSON，不重新训练；中文结果报告由单独的报告编写与核对步骤完成。论文DOCX的集成/排版脚本位于论文修订目录，不是实验重复的必要步骤。

额外锚点检查 `audit/baseline_anchor_check.json` 将R1的M3 BASE与上一轮匹配环境的M3比较，130组概率逐元素一致。该检查依赖前轮APS材料作为历史参照，旧模型/预测未复制进本包；不能仅凭本包重新执行这项历史对照。新实验从原始CSV训练、保存与独立核验不依赖前轮模型。
