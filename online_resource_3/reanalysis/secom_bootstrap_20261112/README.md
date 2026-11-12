# SECOM bootstrap 20,000 次：本次重跑

生成时间：2026-11-12 14:43–14:44（Asia/Shanghai）
工作根：`/home/user/workspace/默认项目/data-analysis/secom_bootstrap_20k_2026-11-12/`

## 这是什么，不是什么

**是**：按原口径（`summarize.py::secom_summary`）重新执行的一次 20,000 次折内分层 bootstrap，四格材料齐全、可复算。

**不是**：2026-11-08 那批 20,000 次运行的原件。那批材料在云电脑上零痕迹（文件名、内容串 `20261108`、首维 20,000 的数组、时间窗全部无命中）。**本目录任何文件都不得在稿件、审稿回复或数据可用性声明里写成「找到原件」或「原始 20,000 次结果」。** 正确表述：

> SECOM 的配对 bootstrap 以折内分层重采样重跑 20,000 次，四组逐次差值与重采样权重已完整保存；重跑前 1,000 次 draw 的汇总值（点估计、区间端点、逐折点估计）与已归档的 1,000 次结果完全一致，确认口径相同。

## 为什么这次重跑可信

1. 千次校验汇总一致：前 1,000 个 draw 复算出的四格点估计、95% 区间端点与五折点估计，与归档 `secom_summary.json` 的 `paired` **差值全为 0.0**。归档没有保存逐次数组，所以这是「汇总一致」，不是「历史逐次数组逐位一致」。
2. 独立实现复算：用 sklearn 的 `average_precision_score(..., sample_weight=)` 重算抽样 draw，与主实现的加权 AP 最大差 ~1e-16。
3. 计数矩阵自洽：每行计数和恒等于该折样本数（314/314/314/313/312），正类计数无一为 0，最大单样本计数 9–10（远低于 uint16 上限）。

校验记录见 `exports/secom_boot20k_validation.json`；独立复核见 `04-independent-verification.md`。

## 目录内容

| 文件 | 内容 |
|---|---|
| `00-intake.md` | 问题、输入版本、grain、冻结口径、交付物、边界 |
| `01-schema.md` | 输入 npz 的键、形状、各折规模、完整性检查 |
| `03-metrics.md` | 四格结果、与归档 1,000 次的对照、逐折离散度、统计纪律 |
| `findings.md` | 结论、可用范围、限制与未决项 |
| `04-independent-verification.md` | 独立复核者（未参与生产）的逐项判定 |
| `scripts/run_secom_boot20k.py` | 全部计算；44.6 秒跑完（含回执包装） |
| `logs/run.log` | 完整 stdout：计数生成、四格逐折、校验门、结果 |
| `logs/execution_receipt.json` | 执行回执：外层命令、cwd、宿主、起止时间、**退出码 0**、脚本与日志 sha256 |
| `logs/harness_receipt/` | **平台终端层回执**（独立于本脚本）：`state=exited`、`exit_code=0`、启动时间、pid、终端输出 |
| `05-rework-log.md` | 独立复核提出的问题与逐条整改证据 |
| `SHA256SUMS.txt` | 工作根清单（不含本报告与被核的 04 复评文件） |
| `scripts/run_with_receipt.sh` | 包装器：运行主脚本并写回执（回执由外层 shell 记 `$?`，不是程序自打印） |
| `exports/secom_boot20k_summary.json` | 四格点估计、区间、分位数、逐折、符号占比、输入哈希 |
| `exports/secom_boot20k_draws.npz` | 四组逐次 ΔAP，形状 (20000, 5)，键名 `{fixed,selected}_C-B__{S0,S1_30}` |
| `exports/secom_boot20k_resample_counts.npz` | 五折各自的逐次重采样计数矩阵，`fold0..fold4`，uint16 (20000, n折) |
| `exports/secom_boot20k_validation.json` | 前 1,000 次与归档的逐值对照（含每折点估计） |
| `exports/secom_boot20k_environment.json` | Python 3.12.3 / numpy 2.5.1 / sklearn 1.9.0、平台、脚本 sha256、参数、起止时间 |

## 怎么读这些数组

```python
import numpy as np

d = np.load("exports/secom_boot20k_draws.npz")
x = d["selected_C-B__S1_30"]      # (20000, 5)：行=draw，列=外层折
print(x.mean(axis=1)[:5])          # 每次 draw 的跨折 ΔAP
print(np.quantile(x.mean(axis=1), [0.025, 0.975]))

c = np.load("exports/secom_boot20k_resample_counts.npz")
cnt = c["fold0"]                   # (20000, 314)：该 draw 里每个测试样本被抽中的次数
w = cnt[0].astype(float)           # 直接就是第 0 次 draw 的权重
```

`schema` 字符串已写进两个 npz，避免日后忘记行列含义。

## 边界

- 随机流固定为 `default_rng(20261101 + 折号)`，与归档 1,000 次运行的构造相同；**这既是校验的基础，也意味着本次不能声称「独立于原运行」**——它是同口径的再执行。
- 四格 95% 区间全部跨 0；这是重采样区间，不是显著性检验，未做多重性校正，也不构成因果主张。
- `draw_mean_mc_se` 是 bootstrap 均值的 Monte Carlo 标准误，不是 AP 差的抽样标准误。
- 每折正类仅 20–21 个，单折估计噪声大，只应看跨折汇总。
- 没有重训任何模型：全部预测来自 `原件补件_2026-11-11/…/secom_fold{0..4}.npz` 已存的逐折概率。
