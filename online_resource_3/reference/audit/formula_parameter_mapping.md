# 公式与协议参数映射

审计对象为第三阶段英文 D 和保留的原生公式。这里给出含义与执行入口；最终 Word 的对象保留由主审计记录检查。所有路径相对项目根目录，历史路径另行完整列出。参数原值完整保存于 protocol_parameter_snapshot.json，未删去未在正文逐项列出的默认模型参数。该快照是当前文件读取，不是历史训练事实的认证。

## 七个原生公式

| 标识与原块 | 数学核对 | 代码或数据入口 | 证据与结论 |
|---|---|---|---|
| main_eq1 MAIN-030 | q(j,y)为该类缺失数除以该类行数，π(y)=n(y)/n | experiments/aps_rerun_20261108/source/ccms.py 的 estimate_class_rates；run_results/split*/preprocessing.npz | 独立从原CSV与训练索引重新估计五拆分的q和类别数，完全一致 |
| main_eq2 MAIN-032 | p_C=α(qmax−q_y)/(1−q_y)，仅原可见格；右侧y=y_i | source/ccms.py:79–90,142–164；run_aps.py:61–69 | 5拆分×3强度×3seed的45个C掩码逐元素一致；q=1跳过、α=0、全空列边界另以玩具矩阵检验 |
| main_eq3 MAIN-034 | E[q′]=(1−α)q+αqmax | 同上；由q+(1−q)p_C直接推出 | 代数成立，执行边界测试成立；只保证期望率收缩，不保证每次实现相同或联合掩码与标签独立 |
| main_eq4 MAIN-037 | q̄=Σ_yπ(y)q(j,y) | source/ccms.py:93–111 | 五拆分按实际类别比例加权；B预算校准仍使用训练标签 |
| main_eq5 MAIN-038 | p_B=α(qmax−q̄)/(1−q̄) | source/ccms.py:93–111,167–184 | 45个B掩码逐元素一致；与C共享U，E[k_C]=E[k_B]=nα(qmax−q̄)，数值误差<1e-10；实际列计数一般不同 |
| main_eq6 MAIN-041 | C−B=(C−A3)+(A3−B)为同一AP尺度恒等式 | run_aps.py:50–69；historical/run_arm3_countmatched.py:348–371 | A3以C实际列计数及seed=ms×1000+7，A4以B实际列计数及seed=ms×1000+11；各15个掩码重建一致。恒等式不使任一分量成为纯机制因果效应 |
| supp_eq1 SUPP-096 | 每次配对重抽先用共享原行权重计算压力输入的AP差，再在压力和五拆分内等权平均 | /Users/bgc/Documents/ChatGPT/research/new/revision/analysis/run_arm3_countmatched.py:144–157,439–446,479–524；SUPP-094–098 | 验证并集unique与searchsorted映射对应φ_s，正负类分别重抽，同一原行跨拆分共享权重；相同得分按阈值组计算加权AP。A3/A4本脚本1000抽，主要三向20000抽另见既有统计核验；不能把条件固定预测区间转给新训练重复 |

预算等式的推导为 Σ_y n_y(1−q_y)α(qmax−q_y)/(1−q_y)=nα(qmax−q̄)，B侧为n(1−q̄)p_B。q_y=1没有可见格，对应项为0；全空列C/B都不删。若α=1，C每类的期望缺失率均为qmax；这不推出掩码向量的联合独立性。

## 当前主要协议与参数

| 协议及英文位置 | 对照、强度和种子 | 模型与聚合 | 直接入口与核对状态 |
|---|---|---|---|
| APS 3.2–3.3 | splits 11/22/33/44/55；α=0/.25/.5/1；mask 9107–9109；A3/A4固定.25 | 1 BASE+18 C/B成员+6 A3/A4=25/拆分，各运行125；HGB auto、100最大迭代；非零ens3；不重拟合 | experiments/aps_rerun_20261108/config.json、run_aps.py:50–96；environment_matched_run/config.json；120非零掩码重建，含BASE125掩码跨环境逐元素相同 |
| 共有选择 3.3 | BASE本组S0−.01可行；四输入S0和S1 10%三seed=101/202/303；最差AP最大；精确同分较小α | α=0复用单个BASE；非零平均3成员概率后计算AP | run_aps.py:83–96；渠道run_channels.py:76起；重抽run_resample.py:58起；两模拟select；液压run.py:39–58；SECOM run_temporal.py:123–134 |
| 压力 3.2 | S0；S1 .1/.3各seed101/202/303；S2随机50%行内60%可见格seed1101/1202/1303；S3约30%列seed2101/2202/2303 | 13输入；S0一个AP，其余各3个AP平均，再等权平均拆分 | 主APS source/run_cb_firstlook.py 的build_eval_inputs；A5同一保存测试掩码；没有把先合并全部压力预测再计算AP |
| 渠道 3.4 | M2 170、M3 340、MO 170维；三workflow重复model2026/2027/2028，mask组19107–19109/29107–29109/39107–39109 | 3×3×5×19=855；auto；同重复同掩码跨渠道；3重复范围非CI | experiments/aps_channels_repeats_20261108/config.json；run_channels.py:34–51,126起；第二阶段保存预测核验复用 |
| A5 3.5 | 首个M3重复、α=.25、mask19107–19109；每C参考5流；seed810000000+split×100000+reference_seed×10+repeat | 每流20E提议含1/2停留及拒绝；75成员、25 ens3单位；5重复范围非CI | experiments/aps_a5_20261108/config.json；run_a5.py:102–165,181–215；本轮75链全提议重放、450检查点、25 ens3重算 |
| A5敏感性 3.5.2 | D由C独立流100E；D seed920000000+split×100000+reference_seed；后续seed930000000+split×100000+reference_seed×10+initial_code×10000000+repeat | C/D×3流×嵌套5/20/50E×5拆分×3参考=270模型；共享D非独立均匀初态；3流范围非CI | experiments/aps_a5_sensitivity_20261109/config.json；run_sensitivity.py:75–96,101–123；本轮15 D初始化+90正式链全重放、285终点/检查点、90 ens3重算 |
| 训练重抽 3.6 | split11；41300负/700正；20 seeds90101–90120；mask19107–19109/model2026；独立anchor | early_stopping=False；固定100迭代；21×19=399；20样本mean/sampleSD/range不含anchor | experiments/aps_training_resample_20261109/config.json；run_resample.py:52–72,153–202；第二阶段保存预测与20次零C选择复核复用 |
| 模拟1 3.7.1 | 601–630、8000×20；信号1.2(X0+X1+X2)，80二分校准均值.05；60/20/20；首6列b=δ=.4，其他b∼U(.05,.30)δ=0 | 20模型/seed、600；model2026/mask9107–9109；m九点1至−1间隔.25；同seed共享X/Y/U | experiments/simulation_expansion_20261108/config.json；run_simulation.py:49–84,107–161；独立30seed及18项Holm家族；20000完整seed向量Bootstrap seed20261108 |
| 模拟2 3.7.2 | 701–730；相同维度与标签信号；Z=clip(X19,−4,4)、斜率1；19列logit类内80二分校准，X19始终观测 | 独立600模型；目标标签仅生成人工目标缺失与评价；另一个18项Holm家族 | experiments/simulation_sensitivity_20261109/config.json；run_simulation.py:56–76,120–178；20000完整seed向量Bootstrap seed20261109；不得混为60seed一批 |
| SECOM 3.8 | 1567×590、104正；约40/60/80%前缀及后20%测试；内验证取前缀末20%；mask9107–9109 | 全590列+590指标；空列中位数0；关闭早停100迭代；先选择后完整前缀重估与19候选重训；3×2×19=114 | revision_v2.4_20261108/experiments/secom_temporal_20261108/config.json；run_temporal.py:33–54,71–138；预测/窗口核验另见阶段四独立分支 |
| 液压 3.9 | 2205周期、17传感器均值/总体SD/min/max=68特征；48组；5外折×4内折首折；训练π校准b=.15/.20/.25/.30及δ=.05/.10/.15/.20 | HGB无早停100迭代；C/B非零ens3；MA罚参0/.001/.01/.1/1/10×depth3/5/7，单树；5×(21+18)=195 | experiments/hydraulic_external_20261109/prepare.py:18–60；run.py:12–58,77–118；MA0/MA_common/MA_tradeoff具体规则与D一致；预测级核验另见阶段四独立分支 |

所有HGB配置的非默认列出值和其余默认值均在参数快照中。各族训练侧预处理只用相应训练分区；SECOM是明确的选择后重拟合例外。其他实验不作选择后重拟合。初始/匹配运行、自动早停/固定迭代、单BASE/非零ens3的差异已在D保留，不把各族汇总量拼接为共同总体推断。

## 解释与信息边界核对

D的摘要、4.3、5.1、6和7对A5的解释一致：固定行列边际后负差仍存在，但无法唯一识别类别分配；D不是全空间独立均匀初态；20E/50E数值接近不证明混合或等效。五个主重复、三个敏感性重复、训练重抽、SECOM窗和液压折均未被错误称为独立总体样本。两种模拟分别定义独立种子与各18项Holm家族，跨零不是等效。

第三阶段原MAIN-108尚有E5正式修复待办，本轮由告警审计分支修复并回填。该事项不影响本文件核查的AP公式、A5预测或预算等式。历史sup-t的跨评价集联合覆盖保证未建立，保留辅助身份；不由当前方法核验声称解决。
