from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
W=Path(__file__).resolve().parents[1];R=W/'results';F=W/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':180,'savefig.dpi':300})
colors={'M3':'#333333','M4':'#0072B2','C025':'#D55E00','B025':'#009E73','C100':'#CC79A7','B100':'#56B4E9','C_selected':'#D55E00','B_selected':'#009E73'}
def read(n):return json.loads((R/n).read_text())
def done(fig,n):
 fig.savefig(F/(n+'.png'),dpi=300);fig.savefig(F/(n+'.pdf'));plt.close(fig)
# 1 Fixed alpha bootstrap; actual complete-family mean, not a representative mask.
x=read('aps_matched_bootstrap.json');fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
labels=['S0','S1 30%','S2','S3']
for tag,offset,color,marker in [('val',-.1,'#0072B2','o'),('test',.1,'#D55E00','s')]:
 for i,k in enumerate(['S0','S1_30','S2','S3']):
  v=x['results'][tag][k];d=v['delta_mean'];lo,hi=v['ci95'];ax.errorbar(d,i+offset,xerr=[[d-lo],[hi-d]],fmt=marker,color=color,capsize=3,label=tag if i==0 else None)
ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Mean AP difference: C(0.25) - B(0.25)');ax.legend(frameon=False);ax.grid(axis='x',alpha=.2);done(fig,'01_APS_same_alpha_paired_intervals')
# 2 Continuous shift, complete target law remains fixed except missingness association.
x=read('simulation_continuous.json');multip=x['target_gap_multipliers'];fig,axs=plt.subplots(1,2,figsize=(7,3.4),layout='constrained')
for ax,cfgs,title in [(axs[0],['M3','M4','C025','B025','C100','B100'],'(a) Fixed alpha'),(axs[1],['M3','C_selected','B_selected'],'(b) Source-selected alpha')]:
 for i,c in enumerate(cfgs):
  a=np.array([[s['metrics'][c][f'gap_{m}'] for m in multip] for s in x['seeds']]);ax.plot(multip,a.mean(0),color=colors[c],marker=['o','s','^','v','D','x'][i],markersize=3,label=c.replace('_selected',' selected'))
 ax.set(title=title,xlabel='Target / source missingness gap',ylabel='AP',ylim=(0,.6));ax.invert_xaxis();ax.grid(alpha=.2);ax.legend(fontsize=7,frameon=False)
done(fig,'02_continuous_missingness_association')
# 3 SECOM independent folds shown, conditional mean intervals are separately tabulated.
x=read('secom_summary.json');fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
for ax,k,title in [(axs[0],'S0','(a) Original'),(axs[1],'S1_30','(b) Extra 30%')]:
 for c in ['M3','M4','C025','B025']:
  v=x['per_config'][c][k]['fold_AP'];ax.plot(range(1,6),v,marker='o',color=colors[c],label=c)
 ax.set(title=title,xlabel='Outer fold',ylabel='AP',xticks=range(1,6),ylim=(0,.27));ax.grid(alpha=.2)
axs[1].legend(frameon=False,fontsize=8);done(fig,'03_SECOM_corrected_per_fold_AP')
# 4 Local evidence and a deletion-budget-controlled localization ablation.
x=read('mechanism_alpha025.json');local=read('localization.json');fig,axs=plt.subplots(1,3,figsize=(7.4,3.1),layout='constrained')
for i,s in enumerate(x['splits']):
 shares=s['pos_deletion_top8_share'];axs[0].scatter([i+1-.12,i+1,i+1+.12],shares,color='#D55E00',s=17)
 b=next(z for z in s['buckets'] if z['observed']==8);axs[1].plot([0,1],[b['mean_score_M3'],b['mean_score_C025']],marker='o',label=str(s['seed']),lw=1)
for i,s in enumerate(local['splits']):
 ds=[]
 for ks in [['S0'],['S1_30_101','S1_30_202','S1_30_303']]:
  ds.append(float(np.mean([s['metrics']['C_local'][k]-s['metrics']['B_local'][k] for k in ks])))
 axs[2].plot([0,1],ds,marker='o',lw=1,label=str(s['seed']))
axs[2].axhline(0,color='black',lw=.8)
axs[0].set(title='(a) Positive deletion share',xlabel='Split',xticks=range(1,6),ylim=(0,1));axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[0].set_ylabel('Share in training-defined top 8')
axs[1].set(title='(b) Fully observed bucket',xticks=[0,1],xticklabels=['M3','C025'],ylabel='Mean score',ylim=(0,.1));axs[1].legend(frameon=False,fontsize=6)
axs[2].set(title='(c) Exact local budget',xticks=[0,1],xticklabels=['S0','S1 30%'],ylabel='AP: C local - B local',ylim=(-.17,.04))
for ax in axs:ax.grid(alpha=.2)
done(fig,'04_actual_intensity_mechanism_localization')
(F/'provenance.json').write_text(json.dumps({'tool':'matplotlib','version':matplotlib.__version__,'sources':['aps_matched_bootstrap.json','simulation_continuous.json','secom_summary.json','mechanism_alpha025.json','localization.json'],'note':'All plotted values come from saved results; no confidence intervals shown on simulation or individual-fold charts.'},indent=2))
print(F)
