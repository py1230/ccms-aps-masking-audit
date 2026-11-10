"""Frozen, complete-case descriptive analysis; never trains or chooses a model."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parent
GROUPS=['S0','S1_10','S1_30','S2','S3']
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def group(k):return k if k=='S0' else k.rsplit('_',1)[0]
def stats(a):
 a=np.asarray(a,float);return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),n=len(a))
def ensemble(paths):
 zs=[np.load(p) for p in paths];y=zs[0]['y'];ids=zs[0]['row_indices'];cs=[k for k in zs[0].files if k not in ['y','row_indices']]
 for z in zs:assert np.array_equal(z['y'],y) and np.array_equal(z['row_indices'],ids)
 probs={k:np.mean([z[k] for z in zs],axis=0) for k in cs};aps={k:float(average_precision_score(y,v)) for k,v in probs.items()}
 return y,ids,probs,aps,{g:float(np.mean([v for k,v in aps.items() if group(k)==g])) for g in GROUPS}
def main():
 assert json.loads((ROOT/'run_results/complete.json').read_text())['fits']==270
 out=ROOT/'analysis';out.mkdir(exist_ok=False);c=json.loads((ROOT/'config.json').read_text());units=[];inputs=[];source=[];timings=[];structures=[];src={}
 for split in c['splits']:
  for arm in ['C','B','BASE']:
   tags=['BASE'] if arm=='BASE' else [f'{arm}025_seed{s}' for s in c['references']]
   _,_,_,aps,ga=ensemble([ROOT/f'reference/split{split}/predictions/{t}__test.npz' for t in tags]);src[split,arm]=ga
   for k,v in aps.items():source.append(dict(split=split,arm=arm,condition=k,AP=v))
  for initial in c['initials']:
   for rep in c['repeats']:
    for step in c['steps']:
     dest=ROOT/f'run_results/split{split}/{initial}/rep{rep}/{step}E'
     y,ids,p,aps,ga=ensemble([dest/f'predictions_ref{rs}.npz' for rs in c['references']]);ed=out/'ensembles'/f'split{split}'/initial/f'rep{rep}';ed.mkdir(parents=True,exist_ok=True);np.savez_compressed(ed/f'{step}E.npz',y=y,row_indices=ids,**p)
     for k,v in aps.items():inputs.append(dict(split=split,initial=initial,repeat=rep,step=step,condition=k,AP=v))
     for g,v in ga.items():units.append(dict(split=split,initial=initial,repeat=rep,step=step,group=g,AP=v,C_minus_A5=src[split,'C'][g]-v,A5_minus_B=v-src[split,'B'][g],A5_minus_BASE=v-src[split,'BASE'][g]))
     for r in json.loads((dest/'fit_records.json').read_text()):timings.append(dict(split=split,initial=initial,repeat=rep,step=step,**r))
 for split in c['splits']:
  for rs in c['references']:
   for initial in c['initials']:
    for rep in c['repeats']:
     z=json.loads((ROOT/f'masks/split{split}/ref{rs}/{initial}/rep{rep}/chain.json').read_text())
     for d in z['diagnostics']:structures.append(dict(split=split,reference=rs,initial=initial,repeat=rep,E=z['E'],**d))
 df=pd.DataFrame(units);df.to_csv(out/'by_split.csv',index=False);pd.DataFrame(inputs).to_csv(out/'by_input.csv',index=False);pd.DataFrame(source).to_csv(out/'source_AP.csv',index=False);pd.DataFrame(timings).to_csv(out/'fit_timings.csv',index=False);pd.DataFrame(structures).to_csv(out/'structural_diagnostics.csv',index=False)
 metrics=['AP','C_minus_A5','A5_minus_B','A5_minus_BASE'];repdf=df.groupby(['initial','step','repeat','group'],sort=False)[metrics].mean().reset_index();repdf.to_csv(out/'by_repeat.csv',index=False)
 summary=[]
 for (initial,step,g),a in repdf.groupby(['initial','step','group'],sort=False):
  for m in metrics:summary.append(dict(initial=initial,step=int(step),group=g,metric=m,**stats(a[m])))
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);dump(out/'summary.json',summary)
 sen=[]
 for g in GROUPS:
  for initial in c['initials']:
   for step in [5,50]:
    a=repdf.query('initial==@initial and group==@g').pivot(index='repeat',columns='step',values='AP');sen.append(dict(group=g,comparison=f'{initial}:{step}E-20E',**stats(a[step]-a[20])))
  for step in c['steps']:
   a=repdf.query('step==@step and group==@g').pivot(index='repeat',columns='initial',values='AP');sen.append(dict(group=g,comparison=f'{step}E:D-C',**stats(a['D']-a['C'])))
 pd.DataFrame(sen).to_csv(out/'sensitivity.csv',index=False);dump(out/'sensitivity.json',sen)
 initcost=sum(json.loads(p.read_text())['seconds'] for p in (ROOT/'masks').glob('split*/ref*/initial_D/chain.json'));contcost=sum(json.loads(p.read_text())['seconds'] for p in (ROOT/'masks').glob('split*/ref*/[CD]/rep*/chain.json'))
 ti=dict(initial_D_generation_seconds=initcost,continuation_generation_seconds=contcost,fit_seconds=sum(r['fit_seconds'] for r in timings),prediction_seconds=sum(r['prediction_seconds'] for r in timings),formal_run=json.loads((ROOT/'run_results/complete.json').read_text()),mask_preparation=json.loads((ROOT/'masks/complete.json').read_text()),smoke=json.loads((ROOT/'smoke/result.json').read_text()))
 dump(out/'timing_summary.json',ti)
 lines=['# A5 步数与初态敏感性实验结果','', '固定完成 270 次正式拟合：五划分、三参考掩码、两初态、三后续随机流及三停止步数。模型拟合前已冻结全部掩码和协议。以下数值先构造 ens3，再平均压力重复与五划分；均值、SD及范围描述三个后续随机流，范围不是置信区间。','', 'D 是每参考掩码由 C 出发运行 100E 次提议生成的一个合法起点，三个后续随机流共享该 D；5E/20E/50E 为同一后续链的嵌套检查点。','', '## 官方 S1 30% 预定主要结果','', '|初态|停止步数|A5 AP 均值|C−A5 均值|C−A5 SD|C−A5 三重复范围|','|---|---:|---:|---:|---:|---|']
 for initial in c['initials']:
  for step in c['steps']:
   get=lambda m:next(s for s in summary if s['initial']==initial and s['step']==step and s['group']=='S1_30' and s['metric']==m)
   a=get('AP');d=get('C_minus_A5');lines.append(f"|{initial}|{step}E|{a['mean']:.6f}|{d['mean']:+.6f}|{d['sd']:.6f}|[{d['minimum']:+.6f}, {d['maximum']:+.6f}]|")
 lines+=['','## 官方 S1 30% 敏感性差','', '|对比|均值|SD|三重复范围|','|---|---:|---:|---|']
 for s in sen:
  if s['group']=='S1_30':lines.append(f"|{s['comparison']}|{s['mean']:+.6f}|{s['sd']:.6f}|[{s['minimum']:+.6f}, {s['maximum']:+.6f}]|")
 lines+=['','全部五压力族的完整结果见 summary.csv、sensitivity.csv。每划分和每测试输入结果分别见 by_split.csv、by_input.csv。','', '## 计时与边界','',f"D 起点生成 {initcost:.2f} 秒；后续链生成 {contcost:.2f} 秒；正式拟合 {ti['fit_seconds']:.2f} 秒；预测 {ti['prediction_seconds']:.2f} 秒。正式进程峰值 RSS {ti['formal_run']['peak_rss_bytes']/2**20:.1f} MiB。接口 smoke 单独记录，不进入正式样本。",'', '该设计只考察给定起点构造、种子与停止步数。各划分重叠且使用共同官方测试集，三个随机流也不是独立数据重复；不计算显著性或等效性，不证明均匀抽样、充分混合、全部合法起点稳健或纯类别因果效应。独立核验结果另见 verification/result.json，哈希与重载检查仅提供内部一致性证据。','']
 (out/'REPORT_中文.md').write_text('\n'.join(lines));dump(out/'complete.json',dict(status='complete',fits=270,ensemble_units=90,group_rows=len(units),input_rows=len(inputs),summary_rows=len(summary),sensitivity_rows=len(sen)))
 print(json.dumps({'status':'complete','output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
