"""Bracket the two >1% original-path slow points without discarding their first samples."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
from resources_gpu3 import snapshot
cases=[dict(name='m1b256_lru_hbm',mtp=1,requests=256,kind='lru',fractions=None,fetch_ctas=16),
 dict(name='m8b64_nocache0_hbm',mtp=8,requests=64,kind='nocache',fractions=[0.],fetch_ctas=24)]
plan=dict(positions=1024,passes=5,settings=['native_hbm'],order=['base','joint_qk','base'],cases=cases,
 base_sha256=sha(p/'base.cuh'),candidate_sha256=sha(p/'joint_qk.cuh'),
 reason='Initialunchangedpaths slow1.113%and1.223%; old34/68exactbutindependentallocationsstillneedbrackets. Preserveoriginalnegativecells.')
file=p/'controls_plan.json'
if file.exists():assert read(file)==plan
else:file.write_text(json.dumps(plan,indent=2)+'\n')
sfile=p/'controls_state.json';state=read(sfile) if sfile.exists() else []
for case in cases:
 for phase,v in [('before','base'),('candidate','joint_qk'),('after','base')]:
  tag='joint_qk_control_'+case['name']+'_'+phase+'_'
  f=p/'gpu3/raw'/f"{tag}{v}_staged_f{case['fetch_ctas']}_i64_soff_{case['kind']}_native_hbm_mtp{case['mtp']}_b{case['requests']}.json"
  if f.exists() and read(f).get('complete'):
   d=read(f);assert d['configuration']['passes']==5 and d['configuration']['positions']==1024
   assert d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==sha(p/(v+'.cuh'))
   print('SKIP_COMPLETED',tag,flush=True);continue
  while not snapshot()['available']:
   print('WAITING_GPU',snapshot(),flush=True);time.sleep(10)
  workspace=(p.parent/'tune_wait_20261031/jit_retained_normal') if v=='base' else p/'jit_joint_qk_normal'
  env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=v,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(workspace),FLASHINFER_EXTRA_CUDAFLAGS='')
  cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),'--kind',case['kind'],'--mtp',str(case['mtp']),'--requests',str(case['requests']),'--settings','native_hbm','--passes','5','--positions','1024','--fetch-ctas',str(case['fetch_ctas']),'--host-split','off']
  if case['fractions'] is not None:cmd+=['--fractions',*map(str,case['fractions'])]
  item=dict(case=case['name'],phase=phase,variant=v,tag=tag,start=time.strftime('%FT%T%z'),command=cmd);state.append(item)
  with (p/(tag+'bench.log')).open('a') as log:
   child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;sfile.write_text(json.dumps(state,indent=2)+'\n');item['exit_code']=child.wait()
  item['end']=time.strftime('%FT%T%z');sfile.write_text(json.dumps(state,indent=2)+'\n')
  if item['exit_code']:raise SystemExit(item['exit_code'])
  print('CONTROL_PHASE_DONE',case['name'],phase,flush=True)
print('CONTROLS_COMPLETE',flush=True)
