# 连续模拟生成器与历史锚点审计

## 核验范围

本报告核对 v2.5 的连续模拟定义、现存源码、现存五种子归档与本次重算。历史文件只读；重算结果不覆盖历史归档，也不证明历史训练的时间、执行者或来源。

源码包根目录（下称 P）：`/Users/bgc/Documents/ChatGPT/research/rr/02_原完整复现包/CCMS论文修订完整资料包_2026-11-01_192149/`。
原模块根目录（下称 S）：`P/03_原始项目归档/CCMS核查_2026-10-31/`。

真正连续模拟入口是 `P/02_本轮代码与结果/scripts/run_supplement.py:119–154` 的 `simulation()`，不是 `S/scripts/sim_nuisance.py` 的 `main()`。后者是早期 80/20、四场景、不同掩码种子的实验；仅其 `gen_data` 和 `base_rates` 被连续模拟入口复用。

## 逐项恢复的协议

|项目|实际实现与定位|
|---|---|
|完整 X|`S/scripts/sim_nuisance.py:30–35,48–63`：8,000×20，`default_rng(data_seed).normal`，各元素标准正态。|
|标签|潜分数 `1.2*(X[:,0]+X[:,1]+X[:,2])`，80次二分法找截距，使该批 X 上 logistic 概率均值约 .05，再从同一 rng 抽8,000个uniform产生伯努利标签。实际正类数随机。|
|原种子|连续入口 `:121` 为 501、502、503、504、505。|
|划分|入口 `:122–126`：rng(data_seed+7)，依次对类别0和1的行号置换，`round(.6*n_c)` 和 `round(.8*n_c)` 处切分，合并后排序。因逐类取整，行数不必恰为4800/1600/1600。|
|基础缺失率|`sim_nuisance.py:66–72`：rng(data_seed+9000)，每列uniform(.05,.30)。连续入口 `:126` 将前6列 b 覆盖为 .4；后14列仍是按数据种子抽出的缺失率。delta前6列=.4，余列=0。|
|自然缺失|入口 `:127–133`：每个数据部分使用其实际类别比例π，q1=b+(1−π)delta*m，q0=b−π*delta*m；无需clip，先断言均在[0,1]。U<q生成掩码。训练与源验证m=1。|
|基础随机数|训练rng(ds*10+1)、源验证rng(ds*10+2)、目标rng(ds*10+3)。九个m对同一目标X/Y与同一U施加阈值；目标完整分布与基础随机数不变。|
|九个目标条件|入口 `:120,135–136`：1、.75、.5、.25、0、−.25、−.5、−.75、−1。|
|源验证压力|入口 `:134`、`S/scripts/run_cb_firstlook.py:49–56`：自然S0和对可见单元额外遮蔽10%的三个S1输入；三个独立U的种子为101、202、303。|
|预处理|入口 `:25–29,138`：中位数仅在自然缺失训练集拟合，所有非零候选共用；M3为中位数填补+全部20列缺失标记，M4保留NaN、不追加标记。全空列中位数置0（`pipeline.py:40–71`）。|
|模型|`pipeline.py:89–91`：HistGradientBoostingClassifier(random_state=2026)，其余使用当时sklearn默认值。当前匹配环境的完整参数见下。|
|干预候选|入口 `:21,139–146`：C和B各α=.25,.5,1；每个候选用9107、9108、9109三个训练掩码种子分别拟合，再平均逐样本概率，之后计算AP。M3、M4各单模型，每数据种子20次拟合。|
|C/B预算|`ccms.py:79–111,142–184`：C条件新增概率α(qmax−qcy)/(1−qcy)，B逐列概率α(qmax−qbar)/(1−qbar)，只删除原可见单元。两者逐列期望删除预算相同，实际删除数不保证相同。同一ms生成同形状同U，跨臂/强度复用。|
|选择|入口 `:31–40,147–148`：每臂候选α=0,.25,.5,1，α=0引用M3。仅源验证参与；S0 AP≥M3 S0 AP−.01为可行；最大化S0与三个S1的最小AP；并列选较小α。|
|归档缺口|入口 `:149–152` 只保存3份索引、y_val/y_test及130个概率数组；不含完整X、y_train、模型、基础U或实际掩码。历史每npz共135数组。|

上述定义与 v2.5 的20维/8000行/约5%/前6列/.4基准率/分层60:20:20/九点m描述一致。新扩展应说明：独立数据种子同时改变完整样本、标签、划分、自然掩码和后14列基础缺失率；不是仅对固定真实总体进行训练集重抽。

## 路径和指纹

|文件|当前SHA-256|
|---|---|
|run_supplement.py|e68bb0f7a05a55d129c66048414b6c4b4aed8d52f98a4a2b5ebfa27fa9fb104a|
|sim_nuisance.py|de9345f5ff238c0de63794fce2e8f31c8129756595498dd06213e42d9179f6cb|
|pipeline.py|45cb6f629e0126f3afd8b1bec4b4b4fb290d0a6888c0c1957cd2202cc0a3a26a|
|ccms.py|2aa211e6ac688c7b1687a1946c30b780a94a3d856a9d0cb59504c0e2554511c2|
|run_cb_firstlook.py|682912992b664996431c9b54c545002e793694680a99fcbf2f12a44c87152809|

历史JSON记录的入口目标哈希为 `90b0fc35f4b122809b6ebb90ce7f461120a45eeef451518d019d8aaf331439d7`，与现入口不同；pipeline目标哈希一致。现入口第11和13行的 `parents[3]` 在当前包布局下指向上一级目录，因此不能原样直接运行。当前核验只提取其matrix/fit_predict/metrics/select函数定义，按明确的现模块目录导入依赖，并逐句恢复simulation主体；该处理保留为本次审计脚本，不能称为找到字节相同的历史入口。

## 环境与初步输入核验

本次实际解释器：`/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`。

实测 Python 3.12.15、NumPy 2.5.4、scikit-learn 1.9.1，与历史simulation_continuous.json记录一致。线程上限2。模型有效默认参数包括：loss=log_loss、learning_rate=.1、max_iter=100、max_leaf_nodes=31、max_depth=None、min_samples_leaf=20、l2_regularization=0、max_bins=255、max_features=1、early_stopping='auto'、validation_fraction=.1、n_iter_no_change=10、tol=1e−7、scoring='loss'、random_state=2026；其余categorical_features='from_dtype'、class_weight=None、interaction_cst=None、monotonic_cst=None、warm_start=False、verbose=0。约4800个训练样本的旧默认auto不触发早停；新代码若显式改为False应注明为参数显式化并核对预测，不能虚称源码字节未改。

所有五种子重新生成后，train/val/test索引和已有y_val/y_test逐元素匹配；训练、源验证和九个目标条件的q1/q0、期望总体缺失率和实际缺失率与历史JSON最大误差为0。重建X的原始float64字节摘要为：

|种子|正类数|X字节SHA-256|
|---|---:|---|
|501|406|13eb6cd78244ce2055f01bfc56c57639f909995a83a2684c536aeb002c9cfb4f|
|502|425|1e681075d819aae23a5998416e4fa8f88cff38f7a312faee4496b245aa6f06f9|
|503|387|755e552bbd91092de8a3994f0d5c8fed3220300f15ee817f611184ed8f84586b|
|504|399|bdee740f01ecf3fb4e95d7da83c58c35077bda01866af9cd10599dd5c9caa1ac|
|505|396|69bddc5137f66b391fb62e3b29a5d02ec5df4fc34b963e27e5389f5e0f371a7d|

这些X是本次再生成值；由于历史归档没有X，不能直接和历史X文件做字节比较。

## 历史模型锚点重算

脚本 `historical_anchor_check.py`，stdout/stderr及 `historical_anchor_check.json` 均保存在本audit目录。命令为上述Python解释器执行 `-u historical_anchor_check.py`。脚本源摘要及导入的源文件摘要写入JSON。此审计预定5×20=100次模型拟合，独立于新正式扩展的拟合计数；详细最终状态以JSON为准。

最终结果：100次历史锚点拟合成功；5种子共650个概率数组逐元素完全一致（最大绝对误差0），450个目标AP误差0，两臂共10项选择及全部候选字典一致，所有索引/标签与缺失元数据检查通过。stderr为空。该结果支持在当前匹配环境下恢复连续模拟的数值行为；它不弥补缺失的历史X原件与入口目标哈希来源链。
