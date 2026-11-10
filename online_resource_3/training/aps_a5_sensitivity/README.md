# APS A5 停止步数与初态敏感性实验

阅读 `PROTOCOL.md` 理解预定设计、依赖结构与解释边界。`freeze.json` 在正式拟合前记录协议、代码、环境、输入、全部初态与掩码摘要；`run_results` 是 270 次真实拟合，`analysis` 是完整后处理，`verification` 是独立核验。原始失败以各阶段的 `failure.json` 保留，输出目录不覆盖。

历史兼容 Python：
`/Users/bgc/Documents/ChatGPT/research/AI论文研究资料包_2026-10-28/ccms_execution/.venv/bin/python`

在全新结果目录重跑时，复制本目录中的代码、`source/`、`config.json`、`input_sources.json`、`raw_data/`、`reference/`、`PROTOCOL.md`、`environment.json` 与 `compiler.json`，不要复制 smoke、masks、freeze.json、run_results、analysis、verification。在相同环境重新编译 `source/switch.c` 为 `source/switch.dylib`，依次执行：

```text
python run_sensitivity.py smoke
python run_sensitivity.py prepare_masks
python run_sensitivity.py freeze
python run_sensitivity.py run
python summarize_results.py
python verify_results.py
```

每个命令保留完整 stdout/stderr。新的运行应具有新的冻结时间与归档目录，不覆盖本次记录。`setup` 仅用于首次从相邻原始 A5 归档复制输入，不在已有配置上重跑。运行器强制环境版本一致、一个进程最多两线程、270 次固定拟合。AP 仅在全部正式拟合完成后计算。

`verification` 用独立 C 实现重放全部提议，并重载 270 个新模型与 35 个源 C/B/BASE 模型，核对全部测试概率及独立 tie-grouped AP；它不重新训练，也不构成历史来源或充分混合认证。
