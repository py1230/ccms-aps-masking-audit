"""Descriptive packaging only, after the complete frozen analysis and verification.

Does not fit, select, change hypotheses or mutate frozen inputs/run artifacts.
"""
from pathlib import Path
from datetime import datetime
import csv, json, hashlib, math, statistics
import numpy as np

HERE = Path(__file__).resolve().parent
def js(p): return json.loads(p.read_text())
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(r): return f"{r['mean']:+.6f} [{r['ci95_t_low']:+.6f}, {r['ci95_t_high']:+.6f}]"

def main():
    summary=js(HERE/'summary.json'); report=js(HERE/'analysis/statistical_summary.json'); ver=js(HERE/'audit/independent_verification.json')
    assert summary['status']=='complete' and summary['fits_completed']==600 and ver['status']=='passed'
    assert ver['counts']['models_reloaded']==600 and len(summary['seeds'])==30
    costs=[]; missing=[]
    for result in summary['seeds']:
        seed=result['data_seed']; root=HERE/'run_results'/f'seed{seed}'
        for r in js(root/'fit_records.json'):
            costs.append({'seed':seed,'model':r['tag'],'stage_seconds':r['seconds'],'n_iter':r['n_iter']})
        for subset,key,m,meta in [('train','train_natural',1,result['train_missing']),('validation','val_natural',1,result['validation_missing'])]+[('test',f'target_{k}',float(k.split('_')[1]),v) for k,v in result['target_missing'].items()]:
            missing.append({'seed':seed,'subset':subset,'mask_key':key,'m':m,'expected_overall_missing':meta['expected_overall_missing'],'actual_overall_missing':meta['actual_missing'],'class_mean_calibration_max_error':meta['class_mean_max_error'],'auxiliary_always_observed':meta['auxiliary_always_observed']})
    csvout(HERE/'analysis/model_stage_costs.csv',costs); csvout(HERE/'analysis/missingness_diagnostics_by_seed.csv',missing)
    elapsed=(datetime.fromisoformat(summary['finished_at_utc'])-datetime.fromisoformat(summary['started_at_utc'])).total_seconds()
    cost={'formal_models':600,'smoke_models_excluded':20,'formal_seeds':30,'automatic_retries':summary['automatic_retries'],
          'failed_formal_attempts':sum(r['return_code']!=0 for r in summary['attempts']),
          'launcher_elapsed_seconds':elapsed,'sum_model_stage_seconds':sum(r['stage_seconds'] for r in costs),
          'median_model_stage_seconds':statistics.median(r['stage_seconds'] for r in costs),
          'worker_processes':1,'native_threads_max':2,'peak_memory_recorded':False,
          'stage_seconds_scope':'from before mask preparation through fit, model serialization, source-validation prediction and saving; not pure fit_seconds',
          'elapsed_scope':'measured launcher start to finish; includes process launch and all per-seed input/output; excludes smoke, freeze, post-run analysis and verification',
          'comparability':'Concurrent experiments can affect elapsed time; no cross-run speed ranking; no extrapolated times or peak memory'}
    dump(HERE/'analysis/cost_summary.json',cost)
    primary=report['primary']; fixed=report['secondary_comparisons'][:9]; selected=report['secondary_comparisons'][9:]
    chosen=report['selection_counts']; rejects=[r for r in fixed+selected if r['reject_holm18_05']]
    overview={'status':'complete_and_verified','protocol':'PROTOCOL.md','freeze':'freeze.json','n':30,'formal_fits':600,'primary':primary,
              'selection_counts':chosen,'holm18_rejected_conditions':[{'comparison':r['comparison'],'m':r['m'],'p_holm18':r['p_holm18']} for r in rejects],
              'verification':ver,'cost':cost,'limits':['Independent second mechanism, not a causal old-versus-new comparison','Expected per-class missingness rates calibrated; realized counts vary','Target labels used only by simulator/metrics; source-only selection','Fixed 30 seeds, no post-result extension','Pointwise intervals, no simultaneous coverage claim','Same-session independent numerical implementation, not independent-team replication']}
    dump(HERE/'analysis/delivery_summary.json',overview)
    rows=['| m | 固定 α=0.25 C−B（95% t区间） | 选择后 C−B（95% t区间） | 固定/选择 Holm p |','|---:|---:|---:|---:|']
    for a,b in zip(fixed,selected): rows.append(f"| {a['m']:g} | {ci(a)} | {ci(b)} | {a['p_holm18']:.4g} / {b['p_holm18']:.4g} |")
    text=f'''# 第二模拟机制：完整运行及核验报告

本轮在用户授权后按本地预先冻结协议完成30个新种子（701–730）、600个正式模型；接口种子99002的20个模型不进入推断。没有按效果扩样、替换种子或重试。冻结文件和原始产物保持不变。本报告生成于全部正式运行及另行实现的核验通过之后。

## 设计与解释范围

完整数据为8000×20，标签真信号仍为1.2(X0+X1+X2)；60/20/20类别分层划分。第二机制把缺失概率设为logistic(a_yj(m)+clip(X19,−4,4))，以80次二分使每类每列平均缺失概率达到预定q_yj(m)。X19始终观测、并不进入真实标签方程。训练与源验证m=1，九个目标m从1到−1共享每种子完整目标X/Y和基础U。保存的最高类别均值校准误差为{max(r['class_mean_calibration_max_error'] for r in missing):.3g}。

该设计在条件于生成矩阵时保留预定每类期望缺失率及总体期望缺失率，改变可观测协变量依赖和联合掩码结构；实际缺失数仍随机。每个子集的Y仅用于模拟缺失生成与评价，预测函数不读取Y。源验证完成并保存选择后才构造目标输入、预测；目标结果不参与参数选择。

这是“标签与已观测协变量共同驱动的缺失”，不无条件称MAR。辅助变量可能帮助模型理解缺失掩码，其始终观测的规定亦有别于旧生成器。因此本次结果是第二机制下的独立检验，不能把与旧30种子结果之差当作机制变化的因果效应，也不合并旧新种子。

## 主要和逐点结果

预定主要估计量是每种子九点等权固定α=0.25 C−B的平均，再在30种子间取均值：**{ci(primary)}**；种子间SD={primary['sd']:.6f}，20,000次完整种子向量Bootstrap的逐点百分位区间为[{primary['bootstrap_percentile_low']:+.6f}, {primary['bootstrap_percentile_high']:+.6f}]。t区间半宽={primary['ci95_t_halfwidth']:.6f}，预定0.005精度目标{'达到' if primary['precision_met'] else '未达到'}。该目标不是等效界限；平均值接近零不能证明各目标条件等效。

{chr(10).join(rows)}

18项次要配对种子检验（9项固定、9项选择后）统一Holm校正，其中{len(rejects)}项达到0.05阈值。t区间与Bootstrap均为逐点区间；主要区间不与这18项家族联合覆盖，不能读作连续m的同时置信带。

固定强度在源机制一致点m=1的差为{fixed[0]['mean']:+.6f}，在m=0为{fixed[4]['mean']:+.6f}，m=−1为{fixed[-1]['mean']:+.6f}，方向随目标缺失条件而变。九点均值会掩盖这种方向差异。选择后的九点中有{sum(r['reject_holm18_05'] for r in selected)}点通过同一Holm家族校正，不能把固定强度某些条件的收益直接转述为完整选择程序的普遍收益。

源验证选择频数（α=0、0.25、0.5、1）：C为{[chosen['C'][str(a)] for a in [0,.25,.5,1]]}，B为{[chosen['B'][str(a)] for a in [0,.25,.5,1]]}。α=0共享M3，选择后不重拟合；0.01的源域容忍阈值为工程规则。

## 留存、成本与核验

每种子目录留存完整X/Y、分层索引、原始U、自然与追加掩码、每行/列/类别×列删除量、20个模型、全部源验证及九点目标概率、候选分数、选择、预处理、参数和事件日志。模型文件是本轮本地生成的可信文件。`manifest.json`记录逐种子文件SHA256，`freeze.json`记录运行前冻结文件SHA256。

自然缺失的概率矩阵另以派生副本保存在`analysis/generation_probabilities/`：这些副本在正式核验后由已存X、类别q和截距重建，没有重新抽样或训练；全部330个自然掩码均由其与原U阈值重建吻合。输入/输出哈希及来源在`analysis/generation_probability_provenance.json`。

另行实现的核验不导入runner或分析代码，以Brent求根复算缺失机制并逐掩码核对，重载{ver['counts']['models_reloaded']}个正式模型、重算{ver['counts']['single_prediction_arrays']}组单模型概率、{ver['counts']['ensemble_prediction_arrays']}组集成概率、{ver['counts']['ap_values']}项AP和{ver['counts']['selections']}次选择。概率最大绝对差={ver['errors']['prediction_max_abs']:.3g}，集成概率差={ver['errors']['ensemble_max_abs']:.3g}，分组并列AP差={ver['errors']['ap_max_abs']:.3g}。20,000次配对Bootstrap索引、主要/次要区间和18项Holm结果均复算通过。该核验与实验在同一会话完成，不称独立团队复现实验，不认证历史训练来源。

正式启动器实测历时{elapsed:.2f}秒；600个模型记录阶段耗时之和为{cost['sum_model_stage_seconds']:.2f}秒。后者包括掩码准备、拟合、模型序列化和源验证预测/保存，**不是纯拟合耗时**。单工作进程、最多2个本地线程；并行的其他实验可能影响耗时，不据此跨运行排名。峰值内存未记录。接口检查、冻结、分析和独立核验耗时不在正式启动器历时内。

## 文件入口

- `PROTOCOL.md`、`config.json`、`freeze.json`：预定机制、模型、推断和冻结哈希。
- `analysis/statistical_summary.json`、`secondary_comparisons.csv`：主要及18项结果。
- `analysis/contrasts_by_seed.csv`、`ap_by_seed.csv`、`selections.csv`：种子级差、AP与选择。
- `analysis/missingness_diagnostics_by_seed.csv`、`model_stage_costs.csv`、`cost_summary.json`：缺失校准与实测成本范围。
- `audit/independent_verification.json`：全部正式产物与统计的复算收据。
- `figures/simulation_second_mechanism_curve.png`及同名SVG/绘图CSV：九点均值与逐点区间；中文图注见`figures/图注.md`。
- `run_results/seed701/`至`seed730/`：全部原始产物；`logs/`保留各次启动日志。

该实验不能替代第三真实数据集或外部方法比较，也不证明对未考察的生成器、标签机制或部署缺失分布具有普遍收益。
'''
    (HERE/'第二模拟机制实验报告.md').write_text(text)
    readme='''# 第二机制模拟实验归档

先阅读[中文报告](第二模拟机制实验报告.md)及[预定协议](PROTOCOL.md)。这是全新的30种子实验，未修改或合并旧实验。`freeze.json`内文件不可修改；已有结果目录禁止覆盖或隐式续跑。

运行环境在`environment.json`和`environment_packages.txt`。正式运行顺序为：编译/边界检查 → `run_simulation.py --smoke` → `verify_results.py --smoke` → `freeze_run.py` → `launch_simulation.py` → `analyze_results.py` → `verify_results.py`。完整运行后另执行`export_generation_probabilities.py`、`report_results.py`、`plot_results.py`生成派生留存、报告和图。当前已完成，不应再次直接运行启动器。`prepare.py`仅是最初构建记录，后续核验器有修正，**不可作为重建全部当前文件的唯一来源**。

`analysis/delivery_summary.json`提供机器可读入口，`audit/independent_verification.json`给出全部核验范围；SHA256只支持文件完整性，不认证其他历史实验。模型仅来自本轮本地拟合。
'''
    (HERE/'README.md').write_text(readme)
    print(json.dumps({'primary':ci(primary),'holm_rejections':len(rejects),'formal_fits':600,'elapsed_seconds':elapsed},ensure_ascii=False))

if __name__=='__main__': main()
