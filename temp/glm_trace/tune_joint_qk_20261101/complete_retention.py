"""Finish already-authorized retention only after live matrix/confirmation work passes its gates."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];jm=p.parent/'tune_joint_mdo_20261101'
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=sha(p/'base.cuh');candidate=sha(p/'retained_joint_qk.cuh');production=root/'include/flashinfer/attention/mla_host_batch.cuh'
state=dict(start=time.strftime('%FT%T%z'),goal_complete=False,candidate_adopted=False,phase='waiting_existing_work',steps=[])
sfile=p/'completion_state.json'
def save():sfile.write_text(json.dumps(state,indent=2)+'\n')
def run(cmd,log,env=None):
 item=dict(command=cmd,log=log,start=time.strftime('%FT%T%z'));state['steps'].append(item)
 with (p/log).open('a') as out:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT);item['pid']=child.pid;save();item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');save()
 if item['exit_code']:raise RuntimeError('Step failed: '+log+' exit'+str(item['exit_code']))
py='/mnt/data/mhy-vllm-venv/bin/python'
def done_plan(folder,tag,subdir):
 plan=read(folder/(tag+'plan.json'))
 for task in plan['tasks']:
  for setting in task['settings']:
   f=folder/subdir/f"{tag}{task['variant']}_staged_f{task['fetch_ctas']}_i64_soff_{task['kind']}_{setting}_mtp{task['mtp']}_b{task['requests']}.json"
   if not f.exists() or not read(f).get('complete'):return False
 return True
save()
while True:
 missing=[]
 for folder,tag in [(jm,'joint_shapes_sxm_limited_'),(jm,'joint_shapes_sxm_native_'),(p,'joint_qk_shapes_sxm_limited_'),(p,'joint_qk_shapes_sxm_native_')]:
  if not done_plan(folder,tag,'raw'):missing.append(str(folder.name)+'/'+tag)
 conf=read(p/'sxm_confirmation_plan.json')
 for cohort in conf['stages']:
  for v in cohort['variants']:
   for setting in cohort['settings']:
    f=p/'raw'/f"{cohort['tag']}{v}_staged_f24_i64_soff_lru_{setting}_mtp8_b128.json"
    if not f.exists() or not read(f).get('complete'):missing.append(cohort['tag']+v+'/'+setting)
 if not missing:break
 # Every wait is associated with a currently live existing process, inspected through /proc.
 processes=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
 live=[line.strip() for line in processes if 'python -u' in line and any(str(folder/name) in line or str((folder/name).relative_to(root)) in line for folder,name in [(jm,'validate_sxm.py'),(p,'validate_sxm.py'),(p,'confirm_sxm.py')]) and '/usr/bin/bash' not in line]
 state.update(missing=missing,live_existing_processes=live,last_observation=time.strftime('%FT%T%z'));save()
 if not live:raise RuntimeError('Missing coverage and no existing producer live; inspect original failures, do not auto-restart.')
 print('WAITING_VERIFIED_EXISTING_WORK',len(missing),'live',len(live),flush=True);time.sleep(20)
state['phase']='final_audit';save()
try:
 run([py,str(jm/'audit_performance.py')],'completion_jm_audit.log')
 run([py,str(jm/'audit_coverage.py')],'completion_jm_coverage.log')
 assert read(jm/'coverage.json')['sxm_matrix_complete']
 for name in ['audit_performance.py','audit_controls.py','audit_coverage.py','report.py']:
  run([py,str(p/name)],'completion_'+name.removesuffix('.py')+'.log')
 run([py,str(p/'retain_joint_qk.py')],'retention_dry_run.log')
 state['phase']='install';save()
 run([py,str(p/'retain_joint_qk.py'),'--install'],'retention_install.log')
 assert sha(production)==candidate
 state['candidate_adopted']=True;state['phase']='fresh_root_validation';save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',FLASHINFER_WORKSPACE_BASE=str(p/'jit_production_joint_normal'),FLASHINFER_EXTRA_CUDAFLAGS='')
 env.pop('FLASHINFER_FAR_EMU',None)
 run([py,'-u',str(p/'compile_production.py')],'compile_production.log',env)
 run([py,'-u',str(p/'test_production.py')],'test_production.log')
 run([py,str(p/'verify_production_codegen.py'),'--normal'],'verify_production_codegen.log')
 probe_env=dict(env,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3')
 run(['numactl','-N1','-m1',py,'-u',str(p/'probe_production_joint_qk.py')],'probe_production_joint_qk.log',probe_env)
 validation=read(p/'tests_production_gpu3.json');assert validation['complete'] and validation['expected_tests']==109
 record=read(p/'joint_qk_retention.json');record['production_validation']=dict(path=str((p/'tests_production_gpu3.json').relative_to(root)),sha256=sha(p/'tests_production_gpu3.json'),codegen_sha256=sha(p/'production_codegen_normal_summary.json'),dispatch_sha256=sha(p/'dispatch_probe_production_joint_qk.json'))
 (p/'joint_qk_retention.json').write_text(json.dumps(record,indent=2)+'\n')
 state.update(phase='complete',complete=True,candidate_adopted=True,end=time.strftime('%FT%T%z'));save()
 print('RETENTION_AND_FRESH_ROOT_VALIDATION_COMPLETE',candidate,flush=True)
except Exception as exc:
 state.update(phase='failed',complete=False,error=str(exc),end=time.strftime('%FT%T%z'))
 if sha(production)==candidate:
  production.write_bytes((p/'base.cuh').read_bytes());state['rolled_back_to_base']=True;state['candidate_adopted']=False
  f=p/'joint_qk_retention.json'
  if f.exists():
   record=read(f);record.update(candidate_adopted=False,post_install_failed=str(exc),production_rolled_back_sha256=base);f.write_text(json.dumps(record,indent=2)+'\n')
 save();raise
