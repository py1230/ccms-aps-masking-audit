# 任务记录：补表 A3（模型种子敏感性）原始执行文件恢复

建立：2026-11-11。发起人：BGC（Lumi 场次）。
目标：找回生成 `results_seed/` 30 条 AP 记录的原始训练程序与执行证据；不重写、不重跑、不补猜参数。

## 用户要求（原话要点）

- 表 A3 = 模型种子敏感性（早期编号 B5），划分 11/22/33/44/55 × 臂 C/B × 模型种子 2026/2027/2028 = 30 条，目录 `results_seed`，文件名 `s11_C_2026.json`。
- 明确排除：方法中的逐列计数匹配 A3 对照臂不是本目标。
- 禁止：按论文描述重写程序冒充原始代码；重新训练替换历史结果；猜测补填种子/参数/分区/环境。
- 必须区分「原始文件」「历史代码片段」「文字描述」「推断」。
- 交付：新目录复制原件（不改原件）、可下载 ZIP、逐文件说明与原路径/对话出处、逐项缺失清单、SHA-256 清单并确认下载件存在。

## 已核事实（本轮工具回包）

- `新增实验_2026-11-05/` 内：`run_seed_dispersion.py`(1811B)、`seed.log`(519B)、`results_seed/`(30 个 JSON)。JSON 与 seed.log 30/30 AP 一致（差 <1e-4），无漏无错。s11_C_2026.json ap=0.461184996450304 与用户给定值逐位相同。
- `run_new_experiments.py` 模型种子写死 2026（`_make_model(2026)`），输出到 `results/`，四配置 A–D，**不是** A3 程序。已按要求排除。
- 原始创建+启动命令：对话 `785ac274-8ea6-430a-8bb6-d4c7b3deb560`，助手消息 `6464de21-56b3-469b-9c19-a78aad3f9381`，block 121，标题「后台跑训练种子离散度」，block_until_ms=0；命令为 heredoc 写脚本 + `nohup python3 run_seed_dispersion.py > seed.log 2>&1 &` + `echo started`。
- 原始终端回执：`bg_9f8d59e842f8`，started_at 2026-11-04T18:30:01Z，title「后台跑训练种子离散度」，timeout 3600000ms，state=exited，exit_code=0。磁盘元数据在 `~/.lightread/terminals/bg_9f8d59e842f8.{meta,status,process,launch}.json`（meta.json 内嵌的命令被截断到 260 字符，完整命令只能从对话取）。
- 交付链：同对话 block 138 —— `cp run_seed_dispersion.py seed.log /tmp/esm2b/.../new_experiments_2026-11-05/ 2>/dev/null || true`，`cp -r results_seed ...`，重新 zip 得 83M ESM_2.zip，438 文件。
- **包内实际情况（本轮独立核验，与当时自述不符）**：投稿包 v5 内 `ESM_2.zip` 的 `new_experiments_2026-11-05/` 只有 run_new_experiments.py / summarize.py / summary.md / summary.json / run.log + results/ + results_seed/。**没有** run_seed_dispersion.py 与 seed.log。可能原因未核：`cp ... 2>/dev/null || true` 吞掉了失败，或 /tmp/esm2b 解包副本里缺少该目录结构后 cp 落空。
- 全盘 `find / -name run_seed_dispersion.py` 只有工作区这一份；`seed.log` 同样只有一份。
- 依赖代码：`CCMS核查_2026-10-31/src/aps_io.py`(2026-10-31 11:28, 787558c6…)、`src/ccms.py`(10-31 11:07, 2aa211e6…)、`scripts/run_cb_firstlook.py`(10-31 11:48, 6829129…)。`scripts/pipeline.py` 该路径不存在，实际位置待查。
- 数据：`默认项目/aps_failure_training_set.csv`（46.7MB）。
- 环境线索：稿件附录 B 自述补充检查用 scikit-learn 1.9.0，存档主表为 1.9.1。运行时未落盘版本快照。
- 文字描述：任务记录 `.lightread/tasks/CCMS_补实验_2026-11-05.md` §「追加（18:40）」；稿件 `Main_Manuscript_revised.docx` 附录 B 段（"three model seeds (2026–2028)"）；`_paa_build/SI_README.txt` 提及 results_seed 30 条。

## 已恢复内容

见同目录 `历史A3原文件恢复/`：run_seed_dispersion.py、seed.log、results_seed/(30)、原始终端回执 4 件、对话执行片段、SHA-256 清单、说明.md。

## 仍未找到（逐项）

1. pipeline.py（_make_model / MedianImputer 的实现在该路径缺失，需另找）。
2. 运行时 scikit-learn/python/numpy 版本的直接记录（只有稿件文字声明 1.9.0）。
3. 该次运行的逐样本预测、模型对象、掩码矩阵（脚本本身不保存这些，程序层面就不产生）。
4. 完整命令 stdout（seed.log 是 nohup 重定向产物，只有 30 行 print + SEED_DONE）。
5. 生成评价输入 S1_30_101 的隐藏实现只在 build_eval_inputs 里，需读 run_cb_firstlook.py 才能确认构造规则（本轮未展开）。

## 进度

- [x] 文件系统与全盘检索
- [x] 对话记录定位原始创建/启动命令
- [x] 终端回执核验
- [x] 压缩包内嵌核验（发现脚本/日志缺失，与当时自述不符）
- [x] 复制原件、打包、SHA-256 清单
- [x] 说明.md 撰写
- [x] 项目记忆更新
