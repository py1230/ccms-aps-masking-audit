"""Freeze a delivery inventory after all writers have completed."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
assert json.loads((HERE/'audit/independent_verification.json').read_text())['status']=='passed'
assert (HERE/'第二模拟机制实验报告.md').exists()
files={str(p.relative_to(HERE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('archive_manifest.json','.DS_Store')}
manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Delivery inventory, not historical provenance certification','excluded':['archive_manifest.json','__pycache__','.DS_Store'],'files':files,'total_files':len(files),'total_bytes':sum(x['bytes'] for x in files.values())}
(HERE/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'files':manifest['total_files'],'bytes':manifest['total_bytes'],'manifest_sha256':sha(HERE/'archive_manifest.json')}))
