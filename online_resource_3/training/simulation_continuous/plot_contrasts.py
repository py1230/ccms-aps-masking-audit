"""Presentation only; reads frozen-analysis CSVs without changing statistics."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    rows=list(csv.DictReader((HERE/'analysis/secondary_comparisons.csv').open()))
    units=list(csv.DictReader((HERE/'analysis/contrasts_by_seed.csv').open()))
    ms=np.array([1,.75,.5,.25,0,-.25,-.5,-.75,-1])
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
    for ax,fam,prefix,title in zip(axs,['fixed_C025_minus_B025','selected_C_minus_B'],['fixed','selected'],['Fixed strength alpha = 0.25','Source-selected procedure']):
        subset=[r for r in rows if r['comparison']==fam]
        y=np.array([float(r['mean']) for r in subset]);lo=np.array([float(r['ci95_t_low']) for r in subset]);hi=np.array([float(r['ci95_t_high']) for r in subset])
        for r in units:ax.plot(ms,[float(r[f'{prefix}__gap_{m:g}']) for m in ms],color='#9da7b1',alpha=.22,lw=.65)
        ax.errorbar(ms,y,yerr=np.vstack([y-lo,hi-y]),color='#145b89',fmt='o-',capsize=3,lw=1.7,label='Mean and pointwise 95% t CI')
        ax.axhline(0,color='#333333',lw=.8,ls='--');ax.set_xlim(1.08,-1.08);ax.set_xticks(ms)
        ax.set_xlabel('Target missingness-gap multiplier m');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Paired AP difference C minus B');axs[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('30 new independent data seeds (601–630)\nGray lines: seed-level contrasts; intervals are not simultaneous',fontsize=12)
    dest=HERE/'analysis/figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'paired_contrasts.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
