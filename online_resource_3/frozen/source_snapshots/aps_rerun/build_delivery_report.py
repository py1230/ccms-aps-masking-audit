"""Build a factual Chinese delivery report from verified outputs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=read('run_results/summary.json');v=read('verification.json');c=read('comparison_summary.json')
    assert s['status']=='complete' and v['status']=='passed'
    rows=list(csv.DictReader((P/'fixed_and_selected_summary.csv').open(encoding='utf-8-sig')))
    means={(r['dataset'],r['condition_group'],r['estimate']):r for r in rows}
    def val(ds,cond,key,kind):return float(means[(ds,cond,key)][kind])
    changes=read('audit/original_documents_hashes.json')
    originals={p:sha(p)==h for p,h in changes.items()};assert all(originals.values())
    (P/'audit/original_documents_preserved.json').write_text(json.dumps(originals,ensure_ascii=False,indent=2))
    selections=read('selection_comparison.json')
    ch=v['checked'];err=v['max_errors']
    text=['# APS 首次环境运行：结果与差异','',
          '本报告仅记录sklearn 1.7.2首次全量运行及其直接数值比较。后续历史版本环境复跑与归因证据见最终《新旧差异与论文更新建议.md》，不能以此阶段报告代替最终解释。','',
          '日期：2026-11-08。当前交付是新增实验、核验和结果对照；原论文 v2.4 保留。', '',
          f'已完成五个划分的 {s["fits_completed"]} 个模型训练，训练和产物写出总耗时 {s["wall_seconds"]:.1f} 秒（不含后续独立核验）。全部候选、计数匹配对照、模型、掩码和预测已保存。新产物核验通过。本次结果与历史数值是否一致，应根据下面的新旧对照判断，不能由“程序跑通”推定。','',
          '## 1. 核心结果','',
          '下表均为五划分等权平均 AP 差 C−B，正数表示 C 高于 B。每个压力重复先分别计算 AP，再平均。固定强度为 α=.25；选择后 C/B 可选不同强度，因此选择后差值不是等预算分配机制对照。','',
          '|评价集|条件|历史固定 C−B|新固定 C−B|历史选择后 C−B|新选择后 C−B|',
          '|---|---|---:|---:|---:|---:|']
    for ds,label in [('val','验证'),('test','官方测试')]:
        for cond in ['S0','S1_10','S1_30','S2','S3']:
            vals=[val(ds,cond,k,t) for k in ['fixed_C_minus_B','selected_C_minus_B'] for t in ['old_mean','new_mean']]
            text.append(f'|{label}|{cond}|'+ '|'.join(f'{z:+.6f}' for z in vals)+'|')
    text+=['','验证集参与选参，不能当作无偏最终表现；官方测试为此前已接触的回顾性评价集。本次没有计算新的置信区间或显著性检验，不移用旧区间。','',
           '固定强度 S1 30% 的逐划分方向：']
    for ds,label in [('val','验证'),('test','官方测试')]:
        d=[r['ap']['C025'][ds]['S1_30']-r['ap']['B025'][ds]['S1_30'] for r in s['splits']]
        text.append(f'- {label}：{sum(z<0 for z in d)}/5 为负；范围 [{min(d):+.6f}, {max(d):+.6f}]。')
    text+=['','五个划分共享样本，方向计数只描述本组运行，不能作为五个独立数据集的检验。','',
           '## 2. 完整候选选择','',
           '|划分|历史完整 C|本次 C|历史完整 B|本次 B|','|---|---:|---:|---:|---:|']
    for row in selections:
        a=row['arms']
        text.append('|'+str(row['split'])+'|'+ '|'.join(str(a[arm][key]['alpha']) for arm in ['C','B'] for key in ['historical_replayed_complete_choice','new_complete_choice'])+'|')
    text+=['','历史完整选择是从同一份保存候选分数表重放，缺失候选的旧逐样本预测并未因此恢复。本次选择可从所有候选的新预测重新计算。α=0 正常参与竞争；固定非零 C 的损失不能直接推广为完整选择程序同样有害。','',
           '## 3. 计数匹配对照','',
           '|评价集|条件|新 C−A3|新 A3−B|新 A4−B|','|---|---|---:|---:|---:|']
    for ds,label in [('val','验证'),('test','官方测试')]:
        for cond in ['S0','S1_30']:
            text.append(f'|{label}|{cond}|'+ '|'.join(f'{val(ds,cond,k,"new_mean"):+.6f}' for k in ['fixed_C_minus_A3','fixed_A3_minus_B','fixed_A4_minus_B'])+'|')
    text+=['','A3/C、A4/B 的逐列实际删除数精确一致。该匹配没有固定逐行负担、具体单元格或联合掩码；C−A3 不解释为纯类别分配因果效应。A4−B 的点差也不能据此判为显著或等效。','',
           '## 4. 结果核验与新旧差异','',
           f'- 重载模型 {ch["models"]} 个，检查单模型预测数组 {ch["prediction_arrays"]} 组；最大预测误差 {err["prediction"]:.3g}。',
           f'- 重建训练掩码 {ch["training_masks"]} 组、评价掩码 {ch["evaluation_masks"]} 组，计数匹配对照 {ch["count_matched_masks"]} 组。',
           f'- 独立复算集成 AP {ch["ap_cells"]} 项，最大绝对误差 {err["ap"]:.3g}；选参核验 {ch["selections"]} 项。',
           f'- 启动冻结文件 {ch["freeze_files"]} 项核验通过；{len(originals)} 份既有 DOCX 指纹保持不变。','',
           f'新旧比较共 {c["cells"]} 个单元：数值一致 {c["status_counts"]["numerically_consistent"]} 项，存在尚未解释的差异 {c["status_counts"]["unexplained_difference"]} 项，缺少历史参照 {c["status_counts"]["missing_historical_reference"]} 项。历史逐样本预测可用 {c["old_prediction_available_cells"]} 项，仅有历史候选分数 {c["old_candidate_score_only_cells"]} 项。1e-12 容差仅判断数值一致，不表示统计等效。','',
           '|配置|可比较单元|数值一致|最大 AP 绝对差|','|---|---:|---:|---:|']
    for name,r in c['by_config'].items():text.append(f'|{name}|{r["comparable"]}|{r["matching"]}|{r["max_abs_ap_difference"]:.6f}|')
    text+=['','已排查的事项：','',
           '- 五个划分的训练/验证行索引逐元素吻合历史保存索引；原始输入哈希一致。',
           '- α=.25 的30组C/B逐列与逐类删除计数和历史记录一致。由于历史未保存完整训练掩码，这不证明逐单元相同。',
           '- 390个历史M3/C025/B025保存预测AP与历史汇总一致（跨旧JSON最大差4.44×10⁻¹⁶），未发现这些重叠数值来源自相矛盾。',
           '- 本次使用未遮蔽训练部分拟合中位数、全部170列和全部缺失指示变量、自动早停、不在选参后重拟合，聚合方式保持历史ens3定义。','',
           '尚未解释或仍缺来源的事项：','',
           '- 本次 Python/NumPy/scikit-learn 为3.13.9/2.3.5/1.7.2；归档记录为3.12.15/2.5.4/1.9.1等不同环境。没有对版本做受控实验，因此环境差仅是候选解释，不能写成已证实原因。',
           '- APS与SECOM的相关生成记录均出现脚本哈希90b0fc35…1439d7，而现存run_supplement.py为e68bb0f7…fb104a；不能将该缺口限定为SECOM。',
           '- 旧稿“C臂未能复现”表述已在v2.2存在；限定范围内未找到对应失败运行原件。旧A3 fidelity零差自述属于另一记录，不据此确认或否认那次失败。',
           '- 官方测试和9,000行预留部分的历史接触范围、历史冻结协议原件及作者声明仍需真实研究记录。','',
           '## 5. 论文处理建议','',
           '1. 将本次结果列为2026-11-08新重跑，独立给出环境、冻结记录与完整候选结果。旧值与新值并排保留，差异尚未定位前，不直接覆盖旧表。',
           '2. 固定强度与完整选择程序分别陈述，依据本次逐划分结果更新稳定性讨论；不能把方向计数或AP点差写成新的显著性结论。',
           '3. 保留计数匹配的解释限制，并把APS生成脚本哈希缺口补入来源局限。模型重载和文件哈希证明本次产物可检查，不证明历史训练。',
           '4. 本任务的训练、保存、核验及差异列示已完成；未解释差异和历史来源问题仍然开放。进入更强机制主张或投稿定稿前，优先定位这些差异，再开展统一信息渠道对照与完整训练流程重复。','',
           '## 6. 文件入口','',
           '- `PROTOCOL.md`、`config.json`、`freeze.json`：设计与固定记录。',
           '- `verification.json`、`run_results/summary.json`：核验与新结果。',
           '- `new_vs_historical.csv`：1170项逐输入对照，缺失历史参照留空。',
           '- `fixed_and_selected_by_split.csv`、`fixed_and_selected_summary.csv`：逐划分及等权均值。',
           '- `selection_comparison.json`、`selection_candidates.csv`：历史与新选择及全部候选分数。',
           '- `audit/`：索引、历史计数、历史数值来源和代码审查。',
           '- `README.md`、`make_repeat.py`：执行和创建独立重复目录的方法。','',
           '本轮由同一会话的助手分工编写、审查和核验，不是盲审、跨模型独立复现或期刊同行评审。生成式AI辅助范围及学术责任应由作者据完整记录如实确认。','']
    (P/'首次环境运行_结果与差异.md').write_text('\n'.join(text),encoding='utf-8')
    print('Report written; original documents preserved:',len(originals))
if __name__=='__main__':main()
