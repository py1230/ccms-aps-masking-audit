# 交付工具静态审查

范围：make_repeat.py、build_package.py、README 和随包汇总入口。仅读取代码及既有 portable_preparation_check.json；没有运行训练、打包或覆盖原件。本报告不是实际 ZIP 核验结果。

## 结论

未发现阻断独立解压后按 README 创建新运行的路径问题。make_repeat 从指定 data-dir 核对原始 CSV 哈希，只修改新目录中的数据绝对路径，核对冻结科学代码并复制 source/historical/scripts；不复制旧结果、旧 smoke 或旧 freeze，也不会启动训练。已有目标目录时直接拒绝。新目录的 freeze_run 会在 smoke 和审查记录存在后生成新冻结与环境快照。

ZIP 有效载荷含原始 CSV、科学代码、模型/掩码/预测、日志和审查材料。manifest 排除自身及外部打包收据；ZIP 内仍保存 manifest 本身且检查其字节一致。压缩包创建后逐成员流式解压、核对完整成员集合、SHA256 和字节数。ZIP 指纹在外部 package_verification.json 中记录，因此没有压缩包或清单哈希自引用。已有 ZIP、manifest 或收据均拒绝覆盖。

## 修订后复查

两项意见已落实并复查：README 准确写为汇总脚本只生成 CSV/JSON，中文报告另行编写与核对；打包器新增 config/freeze/runner/verifier 当前哈希绑定检查，并在存在各渠道报告映射时核对其文件哈希。当前未见其他阻断交付的问题。正式归档仍须等待数值核验和最终报告审查完成；本次没有执行打包。

## 已审查快照

- `experiments/aps_channels_repeats_20261108/make_repeat.py`：`af67c3ea7bd9fa6a9c8a6a953d3b5d9109ac57004a591c2cd4da456e4bffff31`
- `experiments/aps_channels_repeats_20261108/build_package.py`：`f9ab4fe42c10cc3f7a418182984911adf5027cdba3365e1c89d891d45157a18f`
- `experiments/aps_channels_repeats_20261108/README.md`：`d887331e08e8103c6cce61d75a067b4ab8f05a1d5cb95d63a589f9dbb46d1732`
- `experiments/aps_channels_repeats_20261108/scripts/summarize_channels.py`：`c641feac2aed5f0a31d3bb59ef0b87510c5ce5be981a90d875a7a3017e733767`
