"""Launch three paired channels, monitor process exits and preserve logs."""
import json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert (HERE/'freeze.json').is_file()
assert not (HERE/'launcher_receipt.json').exists()
channels=['M2','M3','MO'];handles=[];procs={};start=time.monotonic()
for channel in channels:
    stdout=(HERE/f'{channel}_stdout.log').open('x');stderr=(HERE/f'{channel}_stderr.log').open('x');handles.extend([stdout,stderr])
    procs[channel]=subprocess.Popen([sys.executable,'-u',str(HERE/'run_channels.py'),'--channel',channel],cwd=HERE,stdout=stdout,stderr=stderr)
while True:
    snap={}
    for ch,p in procs.items():
        count=0;path=HERE/'run_results'/ch/'events.jsonl'
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:
                        e=json.loads(line)
                        count=max(count,e.get('fits_completed',0))
                    except json.JSONDecodeError:pass
        snap[ch]={'pid':p.pid,'returncode':p.poll(),'fits_completed':count}
    receipt={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'channels':snap}
    (HERE/'launcher_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    if all(p.poll() is not None for p in procs.values()):break
    time.sleep(30)
for f in handles:f.close()
assert all(p.returncode==0 for p in procs.values()), 'A channel failed; outputs and logs preserved'
print('ALL_CHANNELS_COMPLETED',flush=True)
