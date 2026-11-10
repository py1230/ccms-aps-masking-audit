"""Display all frozen comparisons; no fitting, selection, or interval inference."""
from pathlib import Path
import json,os,platform,sys
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'analysis/mplconfig')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 assert json.loads((ROOT/'analysis/complete.json').read_text())['status']=='complete'
 df=pd.read_csv(ROOT/'analysis/by_repeat.csv');fig,axes=plt.subplots(2,3,figsize=(10.6,6.4),sharex=False);colors={'C':'#176a9a','D':'#cb6a23'}
 for ax,g,label in zip(axes.flat,['S0','S1_10','S1_30','S2','S3'],['S0','S1 10%','S1 30% (primary)','S2','S3']):
  ax.axhline(0,color='#777777',lw=.7,ls=':');ax.set_title(label,fontsize=11)
  for initial,shift in [('C',-.09),('D',.09)]:
   means=[]
   for pos,step in enumerate([5,20,50]):
    values=df.query('initial==@initial and step==@step and group==@g')['C_minus_A5'].to_numpy();mean=values.mean();means.append(mean)
    ax.scatter(pos+shift+np.linspace(-.025,.025,3),values,color=colors[initial],s=15,zorder=4,alpha=.7)
    ax.errorbar(pos+shift,mean,yerr=[[mean-values.min()],[values.max()-mean]],fmt='D',color=colors[initial],mfc='white',ms=4,lw=1,capsize=3,zorder=5)
   ax.plot(np.arange(3)+shift,means,color=colors[initial],lw=.8,alpha=.6,label=f'Start {initial}')
  ax.set_xticks([0,1,2],['5E','20E','50E']);ax.set_xlabel('Continuation proposals');ax.set_ylabel('AP(C reference) - AP(A5)');ax.tick_params(labelsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 axes.flat[-1].axis('off');axes.flat[-1].text(.04,.92,'Fixed finite-step sensitivity',fontsize=12,weight='bold',va='top')
 axes.flat[-1].text(.04,.78,'C: reference C mask\nD: one 100E dispersed start per reference\n\nDots: three continuation streams\nDiamonds: mean; bars: min-max\nEach value averages five split-level ens3 APs.\n\nRanges are not confidence intervals.\nNested steps and shared starts are dependent.\nNo uniform-sampling or mixing claim.',fontsize=9,va='top',linespacing=1.5)
 axes.flat[0].legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(ROOT/'analysis/a5_sensitivity.png',dpi=250,bbox_inches='tight');fig.savefig(ROOT/'analysis/a5_sensitivity.pdf',bbox_inches='tight')
 (ROOT/'analysis/figure_environment.json').write_text(json.dumps(dict(python=platform.python_version(),executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,scope='CSV display only; no model loading, fitting, AP recalculation or selection'),indent=2)+'\n')
if __name__=='__main__':main()
