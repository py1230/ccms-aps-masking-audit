"""Prepare and execute an original training family in a new output directory."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, zipfile
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['aps_rerun','aps_channels','aps_resample','aps_a5','aps_a5_sensitivity','simulation_continuous','simulation_feature_dependent','secom_temporal','hydraulic']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def load_module(path):
 sys.path.insert(0,str(path.parent))
 spec=importlib.util.spec_from_file_location('current_workflow',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def verify_data(data_root,dataset):
 rows=json.loads((ROOT/'data/raw_member_fingerprints.json').read_text())[dataset]
 for row in rows:
  path=data_root/dataset/Path(row['file']).name
  if not path.is_file() or sha(path)!=row['sha256']:
   raise ValueError(f'Missing or changed raw member {path}. Use scripts/acquire_data.py.')

def freeze_inputs(work,style):
 files={}
 for path in sorted(work.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts or path.name=='freeze.json':continue
  if path.relative_to(work).parts[0] in ['run_results','units','verification']:continue
  relative=str(path.relative_to(work))
  files[relative]=({'path':str(path),'sha256':sha(path)} if style=='path' else sha(path))
 dump(work/'freeze.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'files':files,
  'scope':'Fresh portable execution inputs before fitting. Historical research and results were already known. This receipt is not historical provenance.'})

def execute(args):
 dest=args.output.resolve()
 if dest.exists():raise FileExistsError(dest)
 work=dest/args.family
 template='aps_rerun_matched' if args.family=='aps_rerun' and args.aps_environment=='matched' else args.family
 shutil.copytree(ROOT/'training'/template,work)
 if template=='aps_rerun_matched':shutil.copytree(ROOT/'training/aps_rerun/historical',work/'historical')
 data_root=args.data_root.resolve()
 family=args.family
 config_path=work/'config.json'
 config=json.loads(config_path.read_text()) if config_path.exists() else {}
 changes={}
 if family.startswith('aps_'):
  verify_data(data_root,'aps')
  raw=work/'raw_data';raw.mkdir()
  for name in ['aps_failure_training_set.csv','aps_failure_test_set.csv']:
   shutil.copy2(data_root/'aps'/name,raw/name)
  for key,name in [('train_csv','aps_failure_training_set.csv'),('test_csv','aps_failure_test_set.csv')]:
   if key in config:config[key]=str(raw/name);changes[key]='relocated verified raw input'
 if family in ['aps_a5','aps_a5_sensitivity']:
  shutil.copytree(ROOT/'training/shared/a5_reference',work/'reference')
 if family=='secom_temporal':
  verify_data(data_root,'secom')
  for key,name in [('data_path','secom.data'),('label_path','secom_labels.data')]:
   config[key]=str(data_root/'secom'/name);changes[key]='relocated verified raw input'
 # Unit selection changes only the execution schedule. Data and per-model settings remain intact.
 if args.unit:
  if family=='aps_rerun':config.update(split_seeds=[11],expected_models=25)
  elif family=='aps_channels':config.update(split_seeds=[11],repeats=config['repeats'][:1],expected_models_per_channel=19,expected_models=19)
  elif family=='aps_a5':config.update(split_seeds=[11],randomization_repeats=[1],expected_fits=3)
  elif family=='aps_a5_sensitivity':config.update(splits=[11],initials=['C'],repeats=[1],steps=[5],expected_fits=3)
  elif family=='secom_temporal':config['execution_windows']=[1]
  changes['execution_schedule']='one prespecified original unit; no observations or model iterations reduced'
 if config:dump(config_path,config)
 dump(work/'portable_preparation.json',{'family':family,'unit_mode':args.unit,'changes':changes,
  'source_template':str((ROOT/'training'/template).relative_to(ROOT)),
  'aps_environment':args.aps_environment if family=='aps_rerun' else None,
  'original_config_sha256':sha(ROOT/'training'/template/'config.json') if config else None})
 if family=='hydraulic':
  verify_data(data_root,'hydraulic')
  (work/'data').mkdir()
  archive=work/'data/uci447_legacy.zip'
  source_archive=data_root/'hydraulic/uci447_legacy.zip'
  if source_archive.exists():shutil.copy2(source_archive,archive)
  else:
   with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for row in json.loads((ROOT/'data/raw_member_fingerprints.json').read_text())['hydraulic']:
     name=Path(row['file']).name;z.write(data_root/'hydraulic'/name,name)
  shutil.copytree(ROOT/'training/external_baseline_20261109',dest/'external_baseline_20261109')
  dump(work/'execution_schedule.json',{'folds':[1] if args.unit else [1,2,3,4,5]})
  subprocess.run([sys.executable,str(work/'prepare.py')],check=True)
  module=load_module(work/'run.py');module.freeze();module.run();return work
 if family=='aps_a5':
  module=load_module(work/'run_a5.py');module.compile_lib();module.feasibility()
  dump(work/'environment.json',module.env());freeze_inputs(work,'hash')
  for split in config['split_seeds']:module.run(split)
 elif family=='aps_a5_sensitivity':
  subprocess.run(['cc','-O3','-shared','-fPIC',str(work/'source/switch.c'),'-o',str(work/'source/switch.dylib')],check=True)
  module=load_module(work/'run_sensitivity.py');module.prepare_masks();freeze_inputs(work,'hash');module.run()
 elif family in ['simulation_continuous','simulation_feature_dependent']:
  module=load_module(work/'run_simulation.py');freeze_inputs(work,'hash')
  for seed in config['data_seeds'][:1] if args.unit else config['data_seeds']:module.run_seed(seed,config)
 elif family=='aps_rerun':
  module=load_module(work/'run_aps.py');freeze_inputs(work,'path');module.run(config)
 elif family=='aps_channels':
  module=load_module(work/'run_channels.py');freeze_inputs(work,'path')
  for channel in ['M3'] if args.unit else config['channels']:module.run(config,channel)
 elif family=='aps_resample':
  module=load_module(work/'run_resample.py');freeze_inputs(work,'hash')
  for unit in ['anchor'] if args.unit else ['anchor']+[f'seed{s}' for s in config['bootstrap_seeds']]:module.run(unit)
 elif family=='secom_temporal':
  module=load_module(work/'run_temporal.py');freeze_inputs(work,'path');module.run()
 return work

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('family',choices=FAMILIES)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--data-root',type=Path,default=ROOT/'data/raw')
 parser.add_argument('--aps-environment',choices=['matched','first'],default='matched',help='APS primary uses the historical-matched1.9.1 environment. first selects the retained1.7.2 diagnostic rerun.')
 parser.add_argument('--unit',action='store_true',help='One original split/repeat/window/seed with full-size data and unchanged model settings.')
 args=parser.parse_args()
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=2):work=execute(args)
 print(json.dumps({'status':'complete','family':args.family,'unit_mode':args.unit,'work_directory':str(work)}))
if __name__=='__main__':main()
