# 生产方整改记录（针对 04 独立复核）

时间：2026-11-12 15:00（Asia/Shanghai）
被整改对象：`data-analysis/secom_bootstrap_20k_2026-11-12/`
依据：`04-independent-verification.md`（SHA-256 `2a67dc60…`）的 5 条 findings 与 6 条 rework。

**整改后数值未变**：四格 delta_mean 与 ci95 与整改前完全一致（+0.011272 / [−0.012829, +0.030606]；+0.003076 / [−0.010893, +0.019856]；−0.004358 / [−0.028410, +0.020135]；−0.010199 / [−0.023841, +0.002893]）。随机流未动，只删字段、补元数据、改表述。

## 逐条对照

| # | 复核问题（级别） | 处理 | 可核对证据 |
|---|---|---|---|
| 1 | `ci95_transposed` 是对"五个折 bootstrap 均值"取分位，不能当主区间（P1） | **删除该字段**；`draw_mean_se` 更名为 `draw_mean_mc_se` 并附 `draw_mean_mc_se_note`；summary 新增 `ci95_definition` 与 `notes` 三条 | `exports/secom_boot20k_summary.json` 顶层含 `ci95_definition`、`notes`；每格字段表已无 `ci95_transposed`；脚本 `scripts/run_secom_boot20k.py` 中该行已移除 |
| 2 | 缺真实执行命令、cwd、操作系统退出回执（P1） | 新增包装器 `scripts/run_with_receipt.sh`：外层 shell 运行主脚本、捕获 `$?`、写 `logs/execution_receipt.json`；**用它重跑了一次** | 回执含 `inner_command`、`cwd`、`exit_code: 0`（`exit_code_source: outer shell captured $?`）、起止 UTC、`wall_seconds_os: 47`、脚本与日志 sha256、宿主与 Python 版本 |
| 3 | 未能看到生产者在各阶段读取专业指南的时序证据（P1） | **不补造**。本轮实际读取发生在生产对话内（两份 Skill 入口与 5 份必读正文、原 `summarize.py`、原 `run_secom_formal.py`），但无法导出为独立可核证据；复核者镜像只含较早四条消息 | 本条保持"未消除的证据缺口（复核者可见范围限制）"，不作为通过项 |
| 4 | environment 未明示"不是原件"（P2） | 身份补全为"本次重跑（re-run, 2026-11-12）；不是 2026-11-08 那批 20,000 次的原件，原件在云电脑上不存在" | `exports/secom_boot20k_environment.json` 第 2 行 |
| 5 | "逐位一致"表述超出可证范围（P2） | 全部改为"汇总一致（点估计、区间端点、逐折点估计）"，并写明归档未保存逐次数组 | 脚本 docstring 第 6–8 行；`README.md` 第 12、16 行；`findings.md` 第 1、2 条与限制第 2 条；`03-metrics.md` 第 2 节；`00-intake.md` 随机流那一行 |
| 6 | schema 称"四格需要 28 个键"，实际使用 16 个（P2） | 改为"四配置合计 28 个 `__test__` 键全部存在，本口径实际使用 16 个 + `y_test`" | `01-schema.md` 第 3 节 |
| 7 | 区间宽度变化的原因解释无证据（P2） | 撤回"左尾离 0 近、密度变化快"，改为只记录上下限移动事实并注明未做密度/尾部分析 | `03-metrics.md` 第 3 节；`findings.md` 第 3 条 |
| 8 | `default_rng` 跨 NumPy 版本稳定的泛化承诺（P2） | 改为"只在本文件记录的版本内复现；跨版本未测，不作保证" | `environment.json` 的 `numpy_random_note`；`README.md` 复算段 |

## 整改后重跑

- 命令：`bash scripts/run_with_receipt.sh`（回执由包装器写）
- 退出码：0；Python 主脚本 44.6 秒；包装器 OS 计时 47 秒
- 校验门：四格前 1,000 次汇总值与归档差仍为 0.0（`validation.all_pass: true`）
- 产物全部重写（`exports/*`、`logs/run.log`），旧哈希作废，新哈希见 `SHA256SUMS.txt`

## 仍未关闭的项

1. 第 3 条证据缺口（生产时序）——只能由对话记录本身证明，无法在产物内自证。
2. `04-independent-verification.md` 记录的是**整改前**那一版产物哈希；整改后需复核者复评，以新哈希为准。

---

# 第二轮整改（针对复评）

时间：2026-11-12 15:12。依据：复评结论（8 项中 6 项 PASS，2 项 P1、2 项 P3）。

| # | 复评问题 | 处理 | 可核对证据 |
|---|---|---|---|
| 1 | 回执自洽但不能证明当次真实执行（P1） | 改用**平台终端层回执**作为独立来源：把主计算作为后台任务重跑一次，平台自动记下命令、启动时间、进程号与退出状态，这四份文件已归档（非本脚本自写） | `logs/harness_receipt/harness_status.json`：`state=exited`、`exit_code=0`；`harness_meta.json`：完整命令与 `started_at 2026-11-12T07:09:54Z`；`harness_process.json`：pid；`harness_terminal_output.txt`：`EXIT:0` 与 `OUTER_EXIT:0` |
| 2 | schema 总键数写成 121（P3） | 五份 npz 实测均为 **116**，已改 | `01-schema.md` 第 1 节 |
| 3 | `findings.md` 小标题仍用原因式表述（P3） | 改为「区间端点有移动、点估计不变」，与正文不解释原因一致 | `findings.md` 第 3 节 |
| 4 | `05-rework-log.md` 指向的 `SHA256SUMS.txt` 不存在（P3） | 已生成工作根清单（不含本报告与被核的 04 复评文件，因为后者由复核者追加、其哈希会随复评变化；清单内已注明） | `SHA256SUMS.txt` |

## 第二轮后仍未关闭

- **生产准备时序证据缺口（P1）**：复核者只能看到较早的对话镜像，无法核到本轮读取指南与执行检查的原始时序。**不补造、不重跑主计算来充数**，保持未关闭。
- 结论口径：主数值与逐次材料可用；**“完整记录”验收仍为不通过**，原因是上述时序证据缺口，不是数值或产物缺陷。
