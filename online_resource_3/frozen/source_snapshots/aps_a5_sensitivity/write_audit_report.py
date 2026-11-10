"""Package completed numerical outputs and verification for the manuscript editor."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 audit=ROOT.parents[1]/'revision_v3.2_20261109/audit';audit.mkdir(parents=True,exist_ok=True)
 v=json.loads((ROOT/'verification/result.json').read_text());assert v['status']=='passed'
 s=json.loads((ROOT/'analysis/summary.json').read_text());sensitivity=json.loads((ROOT/'analysis/sensitivity.json').read_text());cost=json.loads((ROOT/'analysis/timing_summary.json').read_text());freeze=json.loads((ROOT/'freeze.json').read_text());checks=json.loads((ROOT/'analysis/process_checks.json').read_text())
 bundle={'status':'completed_and_verified','formal_fits':270,'smoke_fits_separate':1,'freeze_timestamp':freeze['timestamp'],'freeze_sha256':sha(ROOT/'freeze.json'),'primary':[r for r in s if r['group']=='S1_30'],'primary_sensitivity':[r for r in sensitivity if r['group']=='S1_30'],'verification':v,'cost':cost,'process_checks':checks,'formal_failure_files':[str(p.relative_to(ROOT)) for p in (ROOT/'run_results').rglob('failure.json')],'figure_png_sha256':sha(ROOT/'analysis/a5_sensitivity.png'),'figure_pdf_sha256':sha(ROOT/'analysis/a5_sensitivity.pdf'),'artifact_directory':str(ROOT)}
 assert not bundle['formal_failure_files']
 (audit/'a5_sensitivity_summary.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 text='''# A5 敏感性实验交付与正文建议

状态：270次正式拟合、统一数值分析和独立核验全部完成。接口 smoke 的一次拟合另列，未计入正式样本。没有正式失败、技术修复重跑、种子替换或方向性扩样。

## 可直接用于方法部分

为考察 A5 的有限步数及初态敏感性，在既有五个 APS 划分与每划分三个 C（α=0.25）参考掩码上，预先设置原 C 和分散初态 D。D 由对应 C 使用独立预定种子运行 100E 次 lazy 交换提议得到，E 为该参考 C 的新增删除格数；每参考只生成一个 D，供三个后续随机流共享。两种初态均精确保留参考 C 的逐行、逐列新增删除计数及原生禁用格。每个后续流保存 5E、20E、50E 三个同链嵌套检查点，所有停留与拒绝均计入提议次数。各检查点均按原 M3/HGB 历史配置重新拟合（model seed=2026、max_iter=100、early_stopping=auto），共5×3×2×3×3=270个新模型；源 C/B/BASE 保留已有匹配模型，不与另一个关闭早停的训练重抽实验混用。

初态、全部检查点掩码、协议及主分析代码在正式拟合前冻结；全部270次拟合完成后才统一计算本轮AP。对同一划分、初态、步数与后续流，先平均三个参考对应模型的概率构造ens3，再分别计算各测试输入AP；同压力族先平均三个压力输入AP，再等权平均五划分，最后报告三个后续流的均值、样本标准差和最小/最大范围。D的100E生成成本单列，不包含在后续5E/20E/50E记号内。范围是条件随机流的描述量，不是置信区间。

## 可直接用于结果与讨论部分

官方 S1 30% 下，C 起点的 A5 平均 AP 在5E、20E和50E时分别为0.71835、0.74436和0.74228；D起点相应为0.74537、0.73968和0.74042。C起点从5E到20E的AP均值增加0.02601，5E时D−C起点差为+0.02703，说明较短运行的步数与起点敏感性不能忽略。20E到50E的均值变化为C起点−0.00208、D起点+0.00075；三个流的差值范围跨零，不能由此作等效性或混合完成判断。六个条件下C−A5的均值均为负（−0.10042至−0.12745），且六条件合计90个“划分×后续流”ens3结果中的该差值均为负，因而原有方向在所检验有限条件内未翻转。

D起点是从C经预定长链生成的一个合法状态，并非从全部可行空间独立均匀抽取；三流共享各自参考的D，三个停止步数同链嵌套，五划分重叠且共享官方测试集。本实验补充了预测层面的步数/初态敏感性证据，不认证均匀抽样、充分混合、全部初态的稳健性或纯类别关联的因果效应。

## 数值表

'''
 text+='\n'.join((ROOT/'analysis/REPORT_中文.md').read_text().splitlines()[2:])
 text+='\n\n## 独立核验与文件身份\n\n'
 text+=f"独立实现重放 {v['chains_checked']} 条链的 {v['all_proposals_replayed']:,} 次提议与 {v['accepted_records_replayed']:,} 条接受记录，核对 {v['checkpoints_checked']} 个检查点（含15个D初态）及全部边际/禁用格。重载 {v['new_models_reloaded']} 个新模型与 {v['source_models_reloaded']} 个源 C/B/BASE 模型，检查 {v['probability_arrays_checked']:,} 个概率数组、{v['probability_values_checked']:,} 个概率值；最大重载误差 {v['max_probability_error']:.3g}。独立 tie-grouped AP、90个ens3、450个划分层结果行、120个汇总行和35个敏感性差汇总均通过；AP最大差异 {v['max_group_AP_error']:.3g}。这是内部一致性验证，不是再次从头训练或历史来源认证。\n\n"
 text+=f"冻结时间（UTC）：{freeze['timestamp']}。freeze.json SHA256：`{bundle['freeze_sha256']}`。本次新模型实际迭代数 {checks['n_iter_min']}–{checks['n_iter_max']}，历史 early_stopping=auto 配置保留。\n\n"
 text+=f"实验完整目录：[{ROOT.name}]({ROOT})。主要机器表为 `analysis/summary.csv`、`analysis/sensitivity.csv`，逐划分/逐输入数据、所有模型/掩码/轨迹/概率及日志均在本目录。代码及重新运行步骤见 `README.md`；图形环境与补充描述检查见 `POSTPROCESSING_NOTES.md`。\n\n"
 text+='## 图注与视觉检查\n\n图：A5在五压力族上的有限步数与初态敏感性。横轴为后续链提议数（E为对应参考C的新增删除格数），纵轴为参考C与A5的AP差；蓝/橙分别为C/D起点。点为三个后续随机流的五划分平均结果，菱形为三流均值，竖线为最小/最大范围，不是置信区间。D另需100E初始化生成；其成本单列。每个流采用相同参考组合的ens3，先平均概率后算AP。各面板纵轴尺度不同。\n\n已打开PNG原图检查：五面板、坐标与标记清楚，无截断重叠，全部压力族均展示。PNG SHA256：`'+bundle['figure_png_sha256']+'`。PDF SHA256：`'+bundle['figure_pdf_sha256']+'`。\n'
 (audit/'a5_sensitivity_results.md').write_text(text)
 (ROOT/'DELIVERY_中文.md').write_text(text)
 print(json.dumps({'status':'written','audit':str(audit/'a5_sensitivity_results.md')},ensure_ascii=False))
if __name__=='__main__':main()
