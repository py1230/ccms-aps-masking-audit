"""Post hoc process and unaggregated direction checks; no extra fits or contrasts."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
def main():
 p=ROOT/'analysis';d=pd.read_csv(p/'fit_timings.csv');a=pd.read_csv(p/'by_split.csv');rows=[]
 for key,b in a.groupby(['initial','step','group']):
  rows.append({'initial':key[0],'step':int(key[1]),'group':key[2],'split_stream_n':len(b),'C_minus_A5_negative_count':int((b.C_minus_A5<0).sum()),'C_minus_A5_min':float(b.C_minus_A5.min()),'C_minus_A5_max':float(b.C_minus_A5.max())})
 pd.DataFrame(rows).to_csv(p/'split_stream_directions.csv',index=False)
 o={'n_iter_min':int(d.n_iter.min()),'n_iter_max':int(d.n_iter.max()),'n_iter_mean':float(d.n_iter.mean()),'all_fits':len(d),'note':'early_stopping=auto historical setting; these are descriptive process checks, not new contrasts'}
 (p/'process_checks.json').write_text(json.dumps(o,indent=2)+'\n')
if __name__=='__main__':main()
