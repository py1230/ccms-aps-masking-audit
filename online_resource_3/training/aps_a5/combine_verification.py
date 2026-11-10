"""Combine distinct mask and model/summary checks without reusing stale evidence."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((HERE/n).read_text())
def main():
 a=read('verification_masks.json');b=read('verification_models.json');align=read('reference_alignment_check.json');freeze=read('freeze.json')
 assert a['status']==b['status']==align['status']=='passed'
 assert a['input_config_sha256']==b['input_config_sha256']==sha(HERE/'config.json')
 assert a['verifier_c_sha256']==b['verifier_c_sha256']==sha(HERE/'verify_switch.c')
 assert b['verifier_sha256']==sha(HERE/'verify_results.py')
 assert b['freeze_sha256']==sha(HERE/'freeze.json')
 for name,h in freeze['files'].items():assert sha(HERE/name)==h,name
 for name,h in b['summary']['file_sha256'].items():assert sha(HERE/name)==h,name
 assert sha(HERE/'mask_structure_summary.json')==b['structure']['source_sha256']
 assert a['masks']['chains']==75 and b['models']['a5_models']==75 and b['models']['reference_models']==35
 assert all(b[k]['status']=='passed' for k in ['models','summary','structure'])
 result={'status':'passed','scope':'independent numeric artifact checks; no independent retraining, uniform mixing proof, historical provenance certification or causal identification','created_utc':datetime.now(timezone.utc).isoformat(),'masks':a['masks'],'models':b['models'],'summary':b['summary'],'structure':b['structure'],'reference_alignment':align,'input_config_sha256':b['input_config_sha256'],'freeze_sha256':b['freeze_sha256'],'check_sources':{n:{'sha256':sha(HERE/n),'check_timestamp':read(n).get('timestamp_utc')} for n in ['verification_masks.json','verification_models.json','reference_alignment_check.json']},'separate_check_note':'Mask verification used the earlier verifier hash recorded in verification_masks.json. Later additions verify summaries/structures; unchanged training freeze checked again at combination.'}
 (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 primary=b['summary']['primary'];m=b['models'];mask=a['masks'];st=b['structure']
 individual=[r['contrasts']['C025-A5']['S1_30'] for r in m['units']]
 lines=['# A5 独立数值与产物核验报告','','## Material Passport','','- artifact_type: independent_artifact_validation','- mode: validate','- Verification Status: ANALYZED（数值与产物核验通过；未独立重训）','- independence: 同一会话中另一助手编写独立代码，不是外部盲审','- scope: 本轮A5、配对参考档案及描述性汇总，不认证历史来源','','## 已完成的检查','',f"75条正式链全部{mask['all_proposals_replayed']:,}次proposal已独立重播，包括接受和拒绝规则、RNG批前后状态/随机数组指纹、接受轨迹和6个检查点。行列边际误差与原生禁用格计数均为0。每个参考5终点两两不同；15个参考均有256个互不共行合法矩形证书，给出至少2^256个合法掩码的构造性下界。下界不证明全空间可达或链混合。",'',f"独立解析原始CSV，重建C/B掩码、中位数、M3输入与13个测试压力输入；全部75个A5训练矩阵/标签指纹、35个参考和75个新模型参数/索引核验通过。重载110个模型核对{m['probability_arrays']:,}个概率数组，最大误差{m['max_prediction_error']:.3g}；25组集成的325个逐压力AP、五组压力与所有差值重算通过，最大AP数值误差{m['max_ap_error']:.3g}。这不是第二次训练。",'',f"35项全局描述性汇总、125行逐单元CSV、25行逐重复CSV及35行摘要CSV独立重算通过，最大误差{b['summary']['max_error']:.3g}。75组结构诊断NPZ/JSON/CSV与类别×列变化范围核验通过。原数据、索引、测试掩码及模型完整参数与配对M3 R1来源一致，115份复制参考文件指纹一致。",'','## 核对后的结果','',f"主要S1 30% C025−A5为{primary['mean']:+.9f} AP；五随机化重复的五划分均值范围[{primary['min']:+.9f}, {primary['max']:+.9f}]。该范围不是置信区间。逐重复×划分的25项差值中负值{sum(v<0 for v in individual)}项；这些相互关联的固定数据结果不作独立符号检验。",'',f"逐类总新增删除量完全保持，但类别×列删除计数L1变化为{st['class_col_l1_range']['min']}–{st['class_col_l1_range']['max']}；前16个活跃列对共删除计数在10E至20E的L1变化为{st['cooccurrence16_10E_20E_range']['min']}–{st['cooccurrence16_10E_20E_range']['max']}。这些诊断不能证明收敛或充分混合。",'','## 解释与统计谬误检查','', '按ARS实验统计解释清单检查11/11项；适用范围为本轮有限步、固定资料的描述性实验。','', '| 项目 | 本轮判断 |','|---|---|',f'| Simpson反转 | 主比较25单元负值{sum(v<0 for v in individual)}项；保留逐单元结果，其他压力分别报告。 |','| 生态谬误 | 不由固定五划分的集成AP推断个体或现场总体效果。 |','| Berkson选择偏差 | APS样本的采集选择机制不能由本实验核实；不外推。 |','| 碰撞变量偏差 | 不作现场因果识别；固定行列边际属于条件算法对照，未证明可忽略选择。 |','| 基率忽略 | 原训练集1000/60000阳性，官方测试375/16000；AP随基率与样本构成变化。 |','| 均值回归 | 未按本轮A5性能筛选参考或种子，不采用极端组前后改进论证。 |','| 幸存者偏差 | 75预定模型全完成，未删失败/改种子；重训未被用于核验。 |','| 多处寻找效应 | 无p值或显著性结论；15组差值与20组绝对AP摘要完整报告。 |','| 分析路径自由度 | 回顾性探索、运行前本地冻结；旧结果已可见，不声称公开预注册。结构附加诊断明确为后验。 |','| 相关即因果 | C−A5仅为固定边际下有限步位置随机化的整体算法差；不识别纯类别分配机制。 |','| 反向因果 | 算法对照顺序明确，但真实缺失与故障的方向未识别，不作现场因果叙述。 |','','## 证据限制','','固定行删除数保留了它与类别的关系；A5仍可改变类别×列和其他联合掩码结构。5次随机化是条件算法重复，五个划分重叠；不覆盖重新抽取训练样本、模型或压力种子变化、外部数据变化及完整alpha选择程序。有限步交换的均匀性、全空间连通性与充分混合均未证明。','', 'verification_masks.json 与 verification_models.json 分别保存检查时间和核验器哈希；verification.json 明确组合两阶段证据。冻结输入在模型核验和组合时再次逐项核对。产物一致性与文件指纹不认证历史训练来源，也不替代作者声明。']
 (HERE/'validation_report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'passed','primary':primary,'prediction_error':m['max_prediction_error'],'ap_error':m['max_ap_error']},ensure_ascii=False))
if __name__=='__main__':main()
