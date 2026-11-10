"""Bounded local runner: three concurrent splits, no automatic retries."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
def stamp():return datetime.now(timezone.utc).isoformat()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def main():
    assert (HERE/'freeze.json').exists()
    assert not (HERE/'launcher_receipt.json').exists()
    cfg=json.loads((HERE/'config.json').read_text())
    waiting=list(cfg['split_seeds']);active={};records=[];failed=False
    receipt={'started_utc':stamp(),'executable':sys.executable,'max_concurrency':3,
             'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'max_split_seconds':cfg['max_run_seconds'],'records':records}
    dump(HERE/'launcher_receipt.json',receipt)
    while active or waiting:
        while waiting and len(active)<3 and not failed:
            s=waiting.pop(0);cmd=[sys.executable,'-u',str(HERE/'run_a5.py'),'--run','--split',str(s)]
            stdout=(HERE/f'split{s}_stdout.log').open('x');stderr=(HERE/f'split{s}_stderr.log').open('x')
            p=subprocess.Popen(cmd,cwd=HERE,stdout=stdout,stderr=stderr)
            rec={'split':s,'pid':p.pid,'command':cmd,'started_utc':stamp(),'status':'running'}
            records.append(rec);active[s]=(p,time.monotonic(),rec,stdout,stderr)
            print(json.dumps(rec),flush=True)
        for s,(p,start,rec,out,err) in list(active.items()):
            if p.poll() is None and time.monotonic()-start>cfg['max_run_seconds']:
                rec['timeout_utc']=stamp();p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            code=p.poll()
            if code is not None:
                out.close();err.close();rec.update(exit_code=code,finished_utc=stamp(),elapsed_seconds=time.monotonic()-start,
                                                status='complete' if code==0 else 'failed')
                print(json.dumps(rec),flush=True);del active[s]
                if code!=0:failed=True
        receipt['queued_unstarted_splits']=waiting
        receipt['updated_utc']=stamp();dump(HERE/'launcher_receipt.json',receipt)
        if failed and not active:break
        if active:time.sleep(5)
    receipt['status']='failed' if failed else 'complete';receipt['finished_utc']=stamp()
    dump(HERE/'launcher_receipt.json',receipt)
    return 1 if failed else 0
if __name__=='__main__':sys.exit(main())
