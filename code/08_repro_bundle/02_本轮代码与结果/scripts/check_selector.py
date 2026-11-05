import run_supplement as r
import numpy as np
conditions=['S0','S1_10_101','S1_10_202','S1_10_303'];y=np.r_[np.zeros(80,dtype=int),np.ones(20,dtype=int)]
good=.1+.8*y;bad=.1+.8*(1-y)
def scenario(base_stress,candidate_stress):
 p={'M3':{k:(good if k=='S0' else base_stress) for k in conditions}}
 for arm in ['C','B']:
  for a in [25,50,100]:p[f'{arm}{a:03d}']={k:(good if k=='S0' else candidate_stress) for k in conditions}
 return r.select(y,p)
a=scenario(good,bad)
assert all(a[c]['chosen']['alpha']==0 for c in ['C','B'])
b=scenario(bad,good)
assert all(b[c]['chosen']['alpha']==.25 for c in ['C','B'])
r.save('selector_regression.json',{'zero_competes_when_nonzero_eligible':'PASS','tie_prefers_smallest_nonzero_when_it_improves':'PASS'})
print('2 selector regression scenarios PASS')
