from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;j=json.loads((HERE/'audit_results.json').read_text());dest=HERE.parents[1]/'revision_v3.2_20261109/figures';dest.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,(a,b)=plt.subplots(1,2,figsize=(9.5,4.8),gridspec_kw={'width_ratios':[1.2,1]})
cs=['S0','S1_10','S1_30','S2','S3'];colors=['#5B6675','#276FBF','#E07A3B','#8E6C8A','#2A9D8F','#929947']
for (arm,label),color,marker in zip([('M3','M3'),('C_selected','C selected'),('B_selected','B selected'),('M4','M4'),('MA0','MA0'),('MA_common','MA common')],colors,['o','s','^','v','D','P']):
 a.plot(range(5),[j['mean_ap'][arm][c] for c in cs],label=label,color=color,marker=marker,linewidth=1.6,markersize=5)
a.set_xticks(range(5),['S0','S1 10%','S1 30%','S2','S3']);a.set_ylabel('Average precision');a.set_ylim(.6,1.01);a.grid(axis='y',alpha=.18);a.set_title('(a) Five-fold mean AP',loc='left',fontweight='bold');a.legend(ncol=3,frameon=False,fontsize=10,loc='lower left')
for comp,label,col,mark in [('C025-B025','C - B, fixed .25','#276FBF','o'),('C_selected-B_selected','C - B, selected','#E07A3B','s'),('MA_common-MA0','MA common - MA0','#2A9D8F','D')]:
 b.plot(range(1,6),j['contrasts'][comp]['S1_30']['fold_differences'],marker=mark,label=label,color=col,linewidth=1.4)
b.axhline(0,color='#6B7280',linestyle='--',linewidth=.8);b.set_xticks(range(1,6));b.set_xlabel('Held-out condition-group fold');b.set_ylabel('Paired AP difference');b.set_ylim(-.07,.075);b.grid(axis='y',alpha=.18);b.set_title('(b) S1 30% fold contrasts',loc='left',fontweight='bold');b.legend(frameon=False,fontsize=10,loc='upper left')
fig.text(.5,.01,'One test rig; imposed missingness. Overlapping training folds; descriptive results.',ha='center',fontsize=9.5,color='#555555')
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(dest/'hydraulic_external.png',dpi=240);fig.savefig(dest/'hydraulic_external.svg');print(dest/'hydraulic_external.png')
