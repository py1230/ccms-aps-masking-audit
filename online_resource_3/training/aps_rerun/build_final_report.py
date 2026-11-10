"""Final report combining both full runs and the bounded runtime intervention."""
from pathlib import Path
import csv
import hashlib
import json

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    Q=P/'environment_matched_run'
    sa,sb=[read(p/'run_results/summary.json') for p in [P,Q]]
    va,vb=[read(p/'verification.json') for p in [P,Q]]
    ca,cb=[read(p/'comparison_summary.json') for p in [P,Q]]
    cross=read(P/'cross_runtime_comparison.json');probe=read(P/'audit/environment_probe/results/summary.json')
    assert va['status']==vb['status']=='passed'
    assert sa['status']==sb['status']=='complete'
    assert cross['scientific_inputs_identical'], 'Resolve input/configuration differences before generating this report.'
    means=[]
    for p in [P,Q]:
        rows=list(csv.DictReader((p/'fixed_and_selected_summary.csv').open(encoding='utf-8-sig')))
        means.append({(r['dataset'],r['condition_group'],r['estimate']):r for r in rows})
    def n(i,ds,cond,kind,field='new_mean'):return float(means[i][ds,cond,kind][field])
    counts=cross['counts'];versions=[read(p/'run_results/environment.json') for p in [P,Q]]
    originals=read(P/'audit/original_documents_hashes.json')
    assert all(sha(f)==h for f,h in originals.items())
    text=['# APS 主分析重跑：最终结果与论文更新建议','',
          '日期：2026-11-08。训练、全候选保存、独立核验、历史数值对照与联合运行环境诊断已完成。原论文 v2.4 保留；本报告不是已投稿稿或历史来源认证。','',
          '## 核心结论','',
          f'本次完成两轮完整五划分运行，各{sa["fits_completed"]}与{sb["fits_completed"]}个模型，另有{probe["n_fits"]}个模型的环境诊断，共{sa["fits_completed"]+sb["fits_completed"]+probe["n_fits"]}次全规模拟合（不含一次小规模接口测试）。',
          f'使用历史记录匹配的关键运行版本后，{counts["historical_probability_arrays"]}组有历史逐样本概率的对照中，{counts["matched_historical_probability_arrays_exact"]}组逐元素相同，最大概率差为{cross["max_errors"]["matched_historical_probability"]:.3g}。可比较的{counts["historical_ap_cells"]}个历史AP单元中，{counts["matched_historical_ap_cells_within_tolerance"]}个在1e-12容差内一致。',
          f'另有{cb["status_counts"]["missing_historical_reference"]}个候选评价单元缺少旧参照，本次已产生并保存新结果，不能称这些单元“复现了旧值”。','',
          f'跨运行的原始数组、科学配置、代码、索引、中位数及全部掩码一致性检查：{cross["scientific_inputs_identical"]}。在本次固定的代码、输入及随机设置下，切换联合运行环境后得到上述与历史一致的概率；未识别Python、NumPy、scikit-learn或其他依赖中哪一个单独造成差异。相同概率属于本次可执行复现证据，不证明历史训练曾按声明发生。','',
          '## 两轮运行及选择规则','',
          '|运行|Python|NumPy|scikit-learn|全量拟合|训练与保存耗时|','|---|---|---|---|---:|---:|']
    for label,env,s in [('首次环境运行',versions[0],sa),('历史版本环境复跑',versions[1],sb)]:
        text.append(f'|{label}|{env["python"]}|{env["numpy"]}|{env["sklearn"]}|{s["fits_completed"]}|{s["wall_seconds"]:.1f}秒|')
    text+=['','两轮使用相同五份历史索引、未遮蔽训练部分拟合的中位数、170列及全部缺失指示变量、显式HGB参数和自动早停。固定模型种子2026，掩码种子9107–9109；没有选参后合并训练与验证集重拟合。所有候选和未选候选均保存模型、掩码与单模型/集成预测。','',
           '完整选择包含α∈{0,.25,.5,1}，验证S0 AP损失不超过.01后最大化S0与三个S1 10%输入的最差AP，并列取较小α。每个划分的选择文件在其测试预测之前保存。','',
           '|划分|历史完整 C|历史环境复跑 C|首次环境 C|历史完整 B|历史环境复跑 B|首次环境 B|',
           '|---|---:|---:|---:|---:|---:|---:|']
    choices=read(Q/'selection_comparison.json')
    first={r['split']:r['chosen_alphas'] for r in sa['splits']}
    for r in choices:
        values=[]
        for arm in ['C','B']:
            a=r['arms'][arm];values += [a['historical_replayed_complete_choice']['alpha'],a['new_complete_choice']['alpha'],first[r['split']][arm]]
        text.append('|'+str(r['split'])+'|'+'|'.join(str(v) for v in values)+'|')
    text+=['','历史完整选择由同一份保存候选分数重放，旧缺失候选预测并未恢复为历史原件；本次新选择则有全部新预测可复算。α=0正常竞争，因此固定非零C的损失不能直接推成完整C程序必然实施非零遮蔽。','',
           f'在相同划分、配置和评价输入的逐单元比较中，AP跨环境最大绝对差为{cross["max_errors"]["ap"]:.6f}；这不是五划分平均差。精确数值与部分B选择对运行环境敏感，应同时报告版本和逐划分结果。','',
           '## 主要点估计','',
           '以下均为五个划分等权平均AP差，正值表示C更高。每个压力输入分别计算AP后再平均。','',
           '|评价集|条件|比较|历史|历史版本环境复跑|首次环境运行|','|---|---|---|---:|---:|---:|']
    for ds,label in [('val','验证'),('test','官方测试')]:
        for cond in ['S0','S1_30']:
            for key,klabel in [('fixed_C_minus_B','固定.25 C−B'),('selected_C_minus_B','选择后C−B'),('fixed_C_minus_A3','固定.25 C−A3')]:
                values=[n(1,ds,cond,key,'old_mean'),n(1,ds,cond,key),n(0,ds,cond,key)]
                text.append(f'|{label}|{cond}|{klabel}|'+ '|'.join(f'{v:+.6f}' for v in values)+'|')
    text+=['','全条件与逐划分结果见两个目录的 `fixed_and_selected_by_split.csv` 和 `fixed_and_selected_summary.csv`。C/B的选择后强度可能不同，选择后差值不是等预算的分配机制对照。','',
           '两个运行环境中固定α=.25、S1 30%的逐划分C−B方向：','']
    for label,s in [('首次环境',sa),('历史版本环境',sb)]:
        for ds,dslabel in [('val','验证'),('test','官方测试')]:
            differences=[r['ap']['C025'][ds]['S1_30']-r['ap']['B025'][ds]['S1_30'] for r in s['splits']]
            text.append(f'- {label}、{dslabel}：{sum(d<0 for d in differences)}/5为负，范围[{min(differences):+.6f}, {max(differences):+.6f}]。')
    text+=['','这些计数描述本组结果，不是独立数据集上的显著性检验。验证集参与选参，官方测试此前已接触；本次均属回顾性描述，没有新置信区间，也不把旧区间贴到首次环境的新预测上。','',
           '## 独立核验范围','',
           '|检查|首次环境|历史版本环境|','|---|---:|---:|']
    for label,key in [('模型重载','models'),('单模型预测数组','prediction_arrays'),('集成AP单元','ap_cells'),('训练掩码','training_masks'),('评价掩码','evaluation_masks'),('计数匹配对照','count_matched_masks'),('选参重放','selections')]:
        text.append(f'|{label}|{va["checked"][key]}|{vb["checked"][key]}|')
    text += [f'|最大重载预测误差|{va["max_errors"]["prediction"]:.3g}|{vb["max_errors"]["prediction"]:.3g}|',
             f'|最大独立AP误差|{va["max_errors"]["ap"]:.3g}|{vb["max_errors"]["ap"]:.3g}|','',
             f'跨环境再次检查{counts["training_masks"]}组训练掩码、{counts["evaluation_masks"]}组评价掩码；具体逐项结果在 `cross_runtime_comparison.json`。原始CSV分别由两个解释器读取后规范化指纹完全一致，排除了本次输入解析数值差异。模型仅在各自版本环境重载，未跨版本加载joblib。','',
             'A3逐列精确匹配C实际删除数，A4匹配B；它们仍不固定行级负担与联合掩码。C−A3不是纯类别分配因果效应，A4−B点差也不能自动解释为显著或等效。','',
             '## 差异定位与仍然开放的来源事项','',
             '首次环境的直接对照CSV先将不同数值标为unexplained_difference；该字段仅描述当时的数值比对。后续环境对照的最终归因范围见 `environment_attribution.csv` 和跨环境JSON，不能只读取阶段报告得出差异仍完全没有解释。','',
             '实际执行中找到了历史fidelity记录中的解释器路径；此前v2.4的可运行环境审计未覆盖该路径。四模型对照先验证split11的M3/C025，之后运行完整五划分及全部候选；两轮都单独冻结，不挑选匹配划分，完整保留不匹配的首次运行。','',
             '仍未恢复的历史材料包括生成记录要求的run_supplement.py目标哈希90b0fc35…1439d7及冻结协议目标哈希95f90d40…7352；现存候选分别为e68bb0f7…fb104a和88481a04…22cb。相关APS与SECOM生成记录均受此限制。限定范围内未找到原件不等于断言原件从未存在。','',
             '旧稿“C臂未能复现”的具体失败运行原件仍未定位；该句已在v2.2存在。本次实测结果应单独报告，不能用那句旧文字概括当前状态，也不能据本次一致性否认过去可能的失败。','',
             '官方测试集与9,000行预留子集的历史使用、作者贡献、AI辅助范围、资助与利益冲突仍需真实作者记录。每环境仅一次固定五划分运行，也不替代独立训练掩码组重抽和重新选择的完整流程稳定性研究。','',
             '## 论文更新与下一步','',
             '1. 在方法与复现说明中固定本次已实测的软件环境、数据索引及完整候选规则；将新125模型及全部候选产物作为本次新增复现证据。',
             '2. 将首次环境运行作为明确的跨环境敏感性结果，保留数值和B选择变化；精确数值应标注对应环境，不能仅写“采用默认参数即可复现”。',
             '3. 继续分开表述固定强度损失、完整程序选择和A3/A4计数对照，保留回顾性设计与机制归因限制。原论文未自动覆盖，具体更新可由本报告逐表对照。',
             '4. 下一实验建议为M2/M3/missingness-only统一信息渠道对照，并加入重新生成训练掩码、模型拟合和重新选参的流程重复。依据目标精度再扩展独立模拟种子；A5按机制主张需要决定。','',
             '## 交付入口','',
             '- `environment_matched_run/`：历史版本环境的完整新训练、预测、核验及新旧对照。',
             '- `run_results/`、`verification.json`、`首次环境运行_结果与差异.md`：首次环境运行与阶段报告。',
             '- `cross_runtime_comparison.json`、`environment_attribution.csv`：跨环境控制变量核对与逐项归因范围。',
             '- `audit/environment_probe/`：先行四模型诊断。`audit/`还含历史来源、索引、计数、原始数组指纹和代码审查。',
             '- `README.md`、`make_repeat.py`：原目录及独立解压后的重复执行方法。',
             '- 上级 `APS主分析重跑完整材料_20261108.zip`：两轮运行、代码、模型、预测、掩码、日志及原始CSV；包内附逐文件清单。','',
             f'{len(originals)}份既有Word文档指纹保持不变。全部核验由同一会话助手分工执行，独立核验指独立代码和计算路径，不是盲审、跨模型独立研究或期刊同行评审。','']
    (P/'新旧差异与论文更新建议.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'report':'新旧差异与论文更新建议.md','historical_arrays_exact':counts['matched_historical_probability_arrays_exact'],'scientific_inputs_identical':cross['scientific_inputs_identical']},ensure_ascii=False))
if __name__=='__main__':main()
