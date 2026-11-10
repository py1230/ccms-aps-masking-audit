"""One verification subprocess at a time; never consume incomplete training units."""
from pathlib import Path
import json,subprocess,sys,time
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent

def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def main():
 cfg=json.loads((HERE/'config.json').read_text());names=['anchor']+[f'seed{s}' for s in cfg['bootstrap_seeds']];history=[]
 root=HERE/'verification_logs';root.mkdir(exist_ok=True)
 while names:
  finished=[n for n in names if (HERE/'verification_units'/f'{n}.json').exists()]
  for name in finished:
   r=json.loads((HERE/'verification_units'/f'{name}.json').read_text());assert r['status']=='passed';names.remove(name)
  if not names:break
  ready=[name for name in names if (HERE/'units'/name/'summary.json').exists() and json.loads((HERE/'units'/name/'summary.json').read_text())['status']=='complete']
  if not ready:
   launcher=json.loads((HERE/'launcher_receipt.json').read_text())
   if launcher.get('status')=='failed':raise RuntimeError('Training failed; verification waits for explicit resolution')
   time.sleep(10);continue
  name=ready[0];started=time.monotonic()
  with (root/f'{name}_stdout.log').open('x') as out,(root/f'{name}_stderr.log').open('x') as err:
   p=subprocess.run([sys.executable,'-u',str(HERE/'verify_results.py'),'--unit',name],stdout=out,stderr=err)
  rec={'unit':name,'exit_code':p.returncode,'seconds':time.monotonic()-started,'finished_utc':datetime.now(timezone.utc).isoformat()};history.append(rec)
  dump(HERE/'verification_worker_progress.json',{'status':'running' if p.returncode==0 else 'failed','records':history,'remaining':names})
  print(json.dumps(rec),flush=True)
  if p.returncode:raise RuntimeError(f'Verifier failed on {name}; no automatic retry')
  names.remove(name)
 dump(HERE/'verification_worker_progress.json',{'status':'complete','records':history,'remaining':[]});print('ALL UNITS VERIFIED',flush=True)
if __name__=='__main__':main()
