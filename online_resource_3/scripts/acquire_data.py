"""Obtain public raw datasets separately and verify historical member fingerprints."""
from pathlib import Path
from urllib.request import urlopen
import argparse, hashlib, json, shutil, zipfile
ROOT=Path(__file__).resolve().parents[1]
URLS={
'aps':'https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip',
'secom':'https://archive.ics.uci.edu/static/public/179/secom.zip',
'hydraulic':'https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=URLS);p.add_argument('--download',action='store_true');p.add_argument('--archive',type=Path,help='Use an already downloaded archive');p.add_argument('--output',type=Path);a=p.parse_args()
 print(json.dumps({'dataset':a.dataset,'url':URLS[a.dataset],'action':'download' if a.download else ('verify local archive' if a.archive else 'plan only'),'license_notice':'Read DATA_ACCESS.md and preserve all original notices.'},indent=2))
 if not a.download and not a.archive:return
 dest=(a.output or ROOT/'data/raw'/a.dataset).resolve();dest.mkdir(parents=True,exist_ok=False)
 if a.archive:archive=a.archive.resolve()
 else:
  archive=dest/'download.zip'
  with urlopen(URLS[a.dataset],timeout=120) as response,archive.open('wb') as f:shutil.copyfileobj(response,f)
 expected=json.loads((ROOT/'data/raw_member_fingerprints.json').read_text())[a.dataset];report=[]
 with zipfile.ZipFile(archive) as z:
  for row in expected:
   options=[n for n in z.namelist() if n==row['file'] or Path(n).name==Path(row['file']).name]
   if len(options)!=1:raise ValueError(f'Expected one member for {row["file"]}; found {options}')
   content=z.read(options[0]);actual=hashlib.sha256(content).hexdigest()
   if actual!=row['sha256']:raise ValueError(f'Fingerprint mismatch for {row["file"]}; preserve and investigate this changed source, do not replace the historical target hash.')
   # Only fingerprint-verified basenames are written, preventing path traversal.
   (dest/Path(row['file']).name).write_bytes(content);report.append({'file':Path(row['file']).name,'sha256':actual,'matched':True})
 (dest/'acquisition_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f'Verified {len(report)} data members. No model fitting was performed.')
if __name__=='__main__':main()
