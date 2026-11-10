"""Post-run display of prespecified estimates; no fitting or statistical decisions."""
from pathlib import Path
import csv,json,math,html
HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'analysis/statistical_summary.json').read_text())
ver=json.loads((HERE/'audit/independent_verification.json').read_text())
assert report['status']=='complete' and ver['status']=='passed'
rows=report['secondary_comparisons']; groups=[rows[:9],rows[9:]]
out=HERE/'figures';out.mkdir(exist_ok=True)
limit=max(abs(r[k]) for r in rows for k in ['ci95_t_low','ci95_t_high'])*1.12
step=max(.002,10**math.floor(math.log10(max(limit,.001))))
limit=math.ceil(max(limit,.006)/step)*step
W,H=1500,810;top,bottom=190,625;lefts=[130,850];width=520
pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 '<rect width="1500" height="810" fill="white"/>',
 '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#243746}.tick{font-size:21px}.label{font-size:24px}.small{font-size:20px}</style>']
def text(x,y,t,cls='label',anchor='start',extra=''):
    pieces.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{html.escape(t)}</text>')
def line(x1,y1,x2,y2,color,width=1.5,extra=''):
    pieces.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}" {extra}/>')
text(750,55,'Second missingness mechanism',anchor='middle',extra='style="font-size:34px;font-weight:bold"')
text(750,96,'30 independent seeds | X19 always observed | class-conditional expected rates calibrated',cls='small',anchor='middle')
for k,(data,left) in enumerate(zip(groups,lefts)):
    color=['#1F5A84','#B9572A'][k]
    def xy(m,y): return left+(1-m)/2*width,top+(limit-y)/(2*limit)*(bottom-top)
    text(left+width/2,151,['(a) Fixed alpha = 0.25','(b) Source-selected alpha'][k],anchor='middle',extra='style="font-size:27px;font-weight:bold"')
    for y in [-limit,-limit/2,0,limit/2,limit]:
        xx,yy=xy(1,y);line(left,yy,left+width,yy,'#8B9CA9' if y==0 else '#DAE1E6',2 if y==0 else 1)
        text(left-14,yy+7,f'{y:+.3f}',cls='tick',anchor='end')
    line(left,top,left,bottom,'#566B79');line(left,bottom,left+width,bottom,'#566B79')
    for r in data:
        x,y=xy(r['m'],r['mean']);a,b=xy(r['m'],r['ci95_t_low'])[1],xy(r['m'],r['ci95_t_high'])[1]
        line(x,a,x,b,color,3);line(x-7,a,x+7,a,color,3);line(x-7,b,x+7,b,color,3)
        pieces.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="6.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        text(x,bottom+34,f"{r['m']:g}",cls='tick',anchor='middle')
    pts=' '.join(f'{xy(r["m"],r["mean"])[0]:.3f},{xy(r["m"],r["mean"])[1]:.3f}' for r in data)
    pieces.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    text(left+width/2,698,'Target class-gap multiplier m (source m = 1)',cls='small',anchor='middle')
text(36,(top+bottom)/2,'Paired AP difference: C minus B',anchor='middle',extra=f'transform="rotate(-90 36 {(top+bottom)/2})"')
p=report['primary']
text(750,754,f"Primary nine-point fixed mean: {p['mean']:+.4f}; 95% t CI [{p['ci95_t_low']:+.4f}, {p['ci95_t_high']:+.4f}]",cls='small',anchor='middle')
text(750,789,'Bars are pointwise 95% seed-level t intervals; they are not simultaneous confidence bands.',cls='small',anchor='middle')
pieces.append('</svg>');(out/'simulation_second_mechanism_curve.svg').write_text('\n'.join(pieces))
keys=['comparison','m','n','mean','ci95_t_low','ci95_t_high','p_nominal','p_holm18','reject_holm18_05']
with (out/'simulation_second_mechanism_curve_data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
(out/'图注.md').write_text('图：第二种缺失机制下C−B的九点曲线。缺失概率由标签与始终观测的辅助变量X19共同驱动，按类别校准每列期望缺失率；每个目标m共享同一数据种子的完整目标X/Y和基础随机数。左：固定α=0.25；右：仅经源验证选择α。点为30个独立数据种子的配对AP差均值，误差线为逐点95% t区间，并非同时置信带。该实验是独立第二机制检验，不将其与旧生成器结果差异作因果归因。\n')
print(out/'simulation_second_mechanism_curve.svg')
