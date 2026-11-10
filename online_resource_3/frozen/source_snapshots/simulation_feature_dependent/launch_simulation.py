"""Bounded local launcher; retains every attempted seed; no retries."""
import concurrent.futures as cf
import json, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent

def stamp():return datetime.now(timezone.utc).isoformat()
def run(ds):
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    start=time.time();error=None
    try:
        with (logs/f'seed{ds}_stdout.log').open('x') as out,(logs/f'seed{ds}_stderr.log').open('x') as err:
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'run_simulation.py'),'--seed',str(ds)],cwd=HERE,stdout=out,stderr=err)
            print(json.dumps({'event':'launched','data_seed':ds,'pid':p.pid,'time':stamp()}),flush=True)
            try:rc=p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                print(json.dumps({'event':'hard_timeout','data_seed':ds,'pid':p.pid,'timeout_seconds':1800}),flush=True)
                p.kill();p.wait();rc=-9;error='hard timeout 1800 seconds'
    except Exception as exc:
        rc=-99;error=repr(exc)+'\n'+traceback.format_exc()
    row={'data_seed':ds,'return_code':rc,'seconds':time.time()-start,'finished_at_utc':stamp(),'error':error,'automatic_retry':False}
    if rc!=0:(logs/f'seed{ds}_failure.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True);return row

def main():
    cfg=json.loads((HERE/'config.json').read_text())
    subprocess.run([sys.executable,str(HERE/'run_simulation.py'),'--check'],check=True)
    assert not (HERE/'run_results').exists() and not (HERE/'launch_results.json').exists(),'Refuse overwrite or implicit restart'
    (HERE/'run_results').mkdir();started=stamp();records=[]
    with cf.ThreadPoolExecutor(max_workers=cfg['workers']) as ex:
        futures={ex.submit(run,ds):ds for ds in cfg['data_seeds']}
        for f in cf.as_completed(futures):
            try:row=f.result()
            except Exception as exc:
                row={'data_seed':futures[f],'return_code':-98,'error':repr(exc),'finished_at_utc':stamp(),'automatic_retry':False}
                (HERE/'logs'/f'seed{futures[f]}_failure.json').write_text(json.dumps(row,indent=2))
            records.append(row);(HERE/'launch_results.json').write_text(json.dumps({'started_at_utc':started,'runs':records},indent=2))
    records.sort(key=lambda z:z['data_seed']);results=[]
    for ds in cfg['data_seeds']:
        p=HERE/'run_results'/f'seed{ds}'/'result.json'
        if p.exists():results.append(json.loads(p.read_text()))
    status='complete' if len(results)==len(cfg['data_seeds']) and all(r['return_code']==0 for r in records) else 'incomplete'
    summary={'status':status,'target_gap_multipliers':cfg['target_gap_multipliers'],'seeds':results,
             'fits_completed':sum(r['fits_completed'] for r in results),'started_at_utc':started,'finished_at_utc':stamp(),
             'attempts':records,'automatic_retries':0}
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({'status':status,'fits_completed':summary['fits_completed']}),flush=True)
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
