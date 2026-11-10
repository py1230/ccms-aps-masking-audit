"""Hash the finished package; no model or analysis mutation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parent
def main():
 assert json.loads((ROOT/'verification/result.json').read_text())['status']=='passed'
 assert (ROOT/'DELIVERY_中文.md').exists()
 files={}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'mplconfig' in p.parts or p.name in ['MANIFEST.json','MANIFEST.sha256']:continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  files[str(p.relative_to(ROOT))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
 obj=dict(status='complete',generated_utc=datetime.now(timezone.utc).isoformat(),file_count=len(files),bytes=sum(v['bytes'] for v in files.values()),scope='all package regular files except self and interpreter/plot caches',files=files)
 (ROOT/'MANIFEST.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'MANIFEST.sha256').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in files.items()))
 print(json.dumps({k:v for k,v in obj.items() if k!='files'}))
if __name__=='__main__':main()
