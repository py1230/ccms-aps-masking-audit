"""Finish disjoint verifier work after training exits, pausing only the scheduler."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def stamp():return datetime.now(timezone.utc).isoformat()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scheduler-pid',type=int,required=True);args=ap.parse_args()
 train=json.loads((HERE/'launcher_receipt.json').read_text());assert train['status']=='complete' and all(r['exit_code']==0 for r in train['records'])
 receipt=HERE/'verification_parallel_receipt.json';assert not receipt.exists()
 cfg=json.loads((HERE/'config.json').read_text());names=['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']]
 state={'started_utc':stamp(),'scheduler_pid':args.scheduler_pid,'status':'starting','records':[]}
 os.kill(args.scheduler_pid,signal.SIGSTOP);state['scheduler_paused_utc']=stamp();dump(receipt,state)
 text=subprocess.check_output(['ps','-Ao','pid=,ppid=,command='],text=True);current=[]
 for line in text.splitlines():
  parts=line.strip().split(None,2)
  if len(parts)==3 and int(parts[1])==args.scheduler_pid and 'verify_results.py --unit ' in parts[2]:current.append(parts[2].split('--unit ',1)[1].strip())
 assert len(current)<=1;state['existing_unit_verifier']=current
 pending=[n for n in names if not (HERE/'verification_units'/f'{n}.json').exists() and n not in current]
 active={};logs=HERE/'verification_logs_parallel';logs.mkdir(exist_ok=False);failed=False
 while pending or active:
  existing_alive=any(not (HERE/'verification_units'/f'{n}.json').exists() for n in current)
  capacity=2 if existing_alive else 3
  while pending and len(active)<capacity and not failed:
   name=pending.pop(0);out=(logs/f'{name}_stdout.log').open('x');err=(logs/f'{name}_stderr.log').open('x')
   p=subprocess.Popen([sys.executable,'-u',str(HERE/'verify_results.py'),'--unit',name],stdout=out,stderr=err)
   record={'unit':name,'pid':p.pid,'started_utc':stamp(),'status':'running'};state['records'].append(record);active[name]=(p,time.monotonic(),out,err,record)
  for name,(p,start,out,err,rec) in list(active.items()):
   code=p.poll()
   if code is not None:
    out.close();err.close();rec.update(exit_code=code,status='complete' if code==0 else 'failed',seconds=time.monotonic()-start,finished_utc=stamp());del active[name];print(json.dumps(rec),flush=True)
    if code:failed=True
  state['status']='running' if not failed else 'failed';state['remaining_unstarted']=pending;dump(receipt,state)
  if failed and not active:break
  if active:time.sleep(5)
 if failed:
  # No failed unit is automatically rerun by the paused sequential scheduler.
  state['scheduler_left_paused_for_diagnosis']=True;dump(receipt,state);return 1
 os.kill(args.scheduler_pid,signal.SIGCONT);state.update(status='complete',scheduler_resumed_utc=stamp(),finished_utc=stamp());dump(receipt,state)
 return 0
if __name__=='__main__':sys.exit(main())
