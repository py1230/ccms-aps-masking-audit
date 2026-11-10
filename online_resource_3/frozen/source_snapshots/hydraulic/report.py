import json,csv,hashlib
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
j=json.loads((HERE/'audit_results.json').read_text());assert j['status']=='passed'
d=json.loads((HERE/'prepared/data_summary.json').read_text());done=json.loads((HERE/'run_results/completion.json').read_text())
labels={'M2':'中位数HGB M2','M3':'中位数+缺失指示M3','M4':'原生NaN HGB M4','C025':'C固定α=.25','B025':'B固定α=.25','C_selected':'C验证选择','B_selected':'B验证选择','MA0':'MA0无惩罚','MA_common':'MA共同AP选择','MA_tradeoff':'MA原思想适配选择'}
cond=['S0','S1_10','S1_30','S2','S3'];out=HERE/'analysis';out.mkdir(exist_ok=True)
with (out/'ap_by_fold.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['fold','arm']+cond)
 for fold in j['folds']:
  for arm in labels:w.writerow([fold['fold'],arm]+[fold['ap'][arm][c] for c in cond])
with (out/'contrasts_by_fold.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['comparison','condition','fold','difference'])
 for comp,vals in j['contrasts'].items():
  for c,v in vals.items():
   for fold,diff in enumerate(v['fold_differences'],1):w.writerow([comp,c,fold,diff])
fits=[]
for fold in range(1,6):
 for r in json.loads((HERE/f'run_results/fold{fold}/fit_records.json').read_text()):fits.append({'fold':fold,**r})
cost={'formal_models':195,'HGB_models':105,'MA_DT_models':90,'failed_formal_fits':0,'automatic_retries':0,'sum_fit_seconds':sum(r['fit_seconds'] for r in fits),'elapsed_seconds':done['elapsed_seconds'],'peak_rss_bytes':done['peak_rss_bytes'],'scope':'fit_seconds is measured fit() only; formal elapsed includes fit, validation/test inference and writes, excludes download, extraction, freezing, smoke, analysis and audit. Concurrent A5 load; no speed ranking.'}
(out/'cost_summary.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2))
lines=['# 独立液压工业数据与MA-DT外部比较','', '**状态：195次正式拟合完成，全部模型、概率、选择、掩码及数据隔离核验通过。** 本协议于2026-11-09性能查看前本地固定，不是历史预注册。原APS/SECOM结果不进入本数据的参数选择。','', '## 材料与设计','', '数据为UCI真实单台架液压试验的2205个周期，严重内部泵泄漏492行（22.31%）。17传感器各取均值、总体标准差、最小与最大值，共68特征；profile标签及稳定标志均排除预测变量。48个非目标工况组合用分层分组外五折、内四折首折验证，三分区组严格互斥。外测行覆盖一次；一个台架、训练折重叠，不是五个独立工业数据集。','', '|外折|训练n/正类|验证n/正类|测试n/正类|训练/验证/测试组数|','|---|---:|---:|---:|---:|']
for r in d['folds']:lines.append(f"|{r['fold']}|{r['train_n']}/{r['train_positive']}|{r['val_n']}/{r['val_positive']}|{r['test_n']}/{r['test_positive']}|{r['train_groups']}/{r['val_groups']}/{r['test_groups']}|")
lines +=['','原数据无缺失。四统计类型固定b=(.15,.20,.25,.30)、δ=(.05,.10,.15,.20)，训练比例π确定q1=b+(1−π)δ、q0=b−πδ；源验证共享该参数，所有分区独立抽掩码。测试基础率为b且不使用标签，再施加与论文一致的S1/S2/S3压力。这是人为标签关联减弱场景，不能称自然缺失/真实设备故障验证。','', 'M2/M3/M4和C/B共105个HGB，统一100迭代、无早停；非零C/B以3模型概率均值形成ens3。作者MA-DT逻辑逐定义AST相同，保留原代码与许可，α={0,.001,.01,.1,1,10}、depth={3,5,7}共90模型。MA0是α0的验证最差AP深度选择；MA_common在同家族S0容忍.01下最大化四输入最差AP；MA_tradeoff按验证S0 AUROC≥最高95%后最小缺失依赖选择。缩小深度网格、统一留出验证及AP选择均是本实验适配，不是原论文完整复现。两类α含义不同。','', '## 完整描述性结果','', '|方法|S0|S1 10%|S1 30%|S2|S3|','|---|---:|---:|---:|---:|---:|']
for a,label in labels.items():lines.append('|'+label+'|'+'|'.join(f"{j['mean_ap'][a][c]:.6f}" for c in cond)+'|')
lines+=['','五折等权AP均值；每折压力输入先分别算AP再取平均；非零C/B先平均模型概率。没有独立折t检验、CI或等效性断言。','', '|S1 30%配对比较|均值|五折范围|逐折差|','|---|---:|---|---|']
for comp,v in j['contrasts'].items():
 r=v['S1_30'];lines.append(f"|{comp}|{r['mean']:+.8f}|[{r['min']:+.6f}, {r['max']:+.6f}]|"+', '.join(f'{x:+.6f}' for x in r['fold_differences'])+'|')
lines+=['','固定C−B为+0.002210，4折正、1折负；选择后为−0.000477，方向混合。新增数据不复制APS固定负向幅度，也不支持普遍正收益。MA_common−MA0的S1 30%均值−0.000015，逐折范围约[−0.0567,+0.0574]；均值接近零不等于等效。HGB与单树MA-DT差距包含模型容量、集成、预处理和网格差异，不能归因于一个缺失处理组件，也不能据此宣称全面优于MA框架。','', '## 选择记录','']
for arm in ['C','B','MA0','MA_common','MA_tradeoff']:lines.append('- '+arm+': '+str(dict(Counter(f['chosen'][arm] for f in j['folds']))))
lines+=['','C：α0一次、.25两次、.5两次；B：α0一次、.5三次、1一次。每折选择在测试预测前写入，所有候选预测均保存，无选择后重拟合。','', '## 核验与成本','',f"冻结129个输入/协议/代码文件，含原ZIP及作者源定义。195模型重载、3315组单模型概率、2295组集成概率、90个C/B追加掩码、130个验证/测试压力输入全部通过。最大概率差0，独立并列分组AP差{j['max_ap_error']:.3g}。同时从ZIP重算全部68特征与标签，核验完整模型集合、参数、100迭代、工况组隔离、预处理只读训练、预算相等及全部选参。",'',f"实测正式历时{cost['elapsed_seconds']:.2f}秒；纯fit()之和{cost['sum_fit_seconds']:.2f}秒；进程峰值RSS{cost['peak_rss_bytes']/2**20:.1f}MiB。单进程、最多2线程，与其他实验并行；不作公平速度排名。下载、特征提取、smoke、冻结、分析及独立核验不包含在正式历时内。零正式失败、零重试。",'', '运行环境Python3.12.15、NumPy1.26.4、SciPy1.13.1、scikit-learn1.5.2、joblib1.4.2；选择作者sklearn版本避免私有接口差异。所有本批HGB及MA共用此环境，不能与其他环境直接作版本归因。','', '## 来源、许可与复现','', '- [UCI数据与CC BY4.0许可](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)，DOI [10.24432/C5CW21](https://doi.org/10.24432/C5CW21)。原ZIP SHA256：'+d['raw_zip_sha256']+'.', '- [MA论文：Stempfle等，ICML2025](https://proceedings.mlr.press/v267/stempfle25a.html)。[作者代码](https://github.com/Healthy-AI/malearn)，commit fbbdd0ca2c9d0d8072f49d82ad35f4391098863f；MIT版权和许可保留在相邻external_baseline_20261109/vendor/malearn/LICENSE.md。15定义抽取与原源码逐字及AST吻合，记录extraction_manifest.json。','- 设计流程使用experimental-design技能辅助，软件引用：Kassis等(2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。它是流程工具披露，不是方法有效性证据。','', '入口：PROTOCOL.md、prepare.py、run.py、freeze.json、audit.py、audit_results.json、prepared/、run_results/、analysis/、data/download_attempts/。本地归档不等于已公开发布；第三方再利用需按数据及代码许可署名。','']
(HERE/'液压数据与外部方法实验报告.md').write_text('\n'.join(lines))
print(json.dumps(cost,ensure_ascii=False,indent=2))
