"""Render frozen diagnostic CSV outputs with Matplotlib; no recomputed inference."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
HERE=Path(__file__).resolve().parent
A=HERE/'analysis'
def read(name):return list(csv.DictReader((A/name).open()))
def title(name):
    if name=='primary':return 'Primary: equal-grid fixed difference'
    family,m=name.split('__gap_');return ('Fixed' if family=='fixed' else 'Selected')+' C - B; m = '+m
names=[r['comparison'] for r in read('normality_diagnostics.csv')]
for which,file,x,y in [('qq','qq_plot_data.csv','normal_quantile','observed'),('histogram','histogram_data.csv','left','count')]:
    rows=read(file);fig,axes=plt.subplots(5,4,figsize=(16,17));axes=axes.ravel()
    for ax,name in zip(axes,names):
        subset=[r for r in rows if r['comparison']==name];xx=np.array([float(r[x]) for r in subset]);yy=np.array([float(r[y]) for r in subset])
        if which=='qq':
            ax.scatter(xx,yy,s=14,color='#176B87');coef=np.polyfit(xx,yy,1);ax.plot(xx,np.polyval(coef,xx),color='#B24C42',linewidth=1);ax.set_xlabel('Theoretical normal quantile');ax.set_ylabel('Ordered seed AP difference')
        else:
            widths=np.array([float(r['right'])-float(r['left']) for r in subset]);ax.bar(xx,yy,width=widths,align='edge',color='#176B87',edgecolor='white');ax.set_xlabel('Seed AP difference');ax.set_ylabel('Seeds')
        ax.set_title(title(name),fontsize=9);ax.grid(alpha=.15);ax.tick_params(labelsize=8);ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if which=='histogram':ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    axes[-1].axis('off');fig.suptitle(('Q-Q plots' if which=='qq' else 'Histograms')+' of 30 independent-seed contrasts\nDescriptive diagnostics; no method switching',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(A/f'{which}_diagnostics.png',dpi=150);fig.savefig(A/f'{which}_diagnostics.pdf');plt.close(fig)
print('Rendered 19 Q-Q plots and 19 histograms from frozen analysis CSVs.')
