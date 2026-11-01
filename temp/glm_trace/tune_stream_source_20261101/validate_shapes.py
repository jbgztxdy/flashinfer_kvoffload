"""Restart-safe local changed-path regression and a small set of unchanged-path controls."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['limited','native'],required=True);parser.add_argument('--plan-only',action='store_true');a=parser.parse_args()
from resources_gpu3 import snapshot
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
v='stream_source'
# Only expand after the successful full-window reverse screen, never from tests alone.
for setting in ['g100_l0','g100_l20000']:
 ds=[read(p/'gpu3/raw'/f'streamsource_reverse1024_f24_{name}_staged_f24_i64_soff_lru_{setting}_mtp8_b128.json') for name in ['base',v]]
 assert all(x['complete'] for x in ds)
 assert ds[1]['cases'][0]['timings']['ours']['mean_us'] < ds[0]['cases'][0]['timings']['ours']['mean_us']
assert read(p/'compiler_stream_source_summary.json')['unchanged_original_kernels']==38
assert read(p/'compiler_stream_source_normal_summary.json')['unchanged_original_kernels']==76
assert read(p/'tests_stream_source_gpu3.json')['complete'] and read(p/'stress_stream_source_gpu3.json')['complete']
if not a.plan_only:
 state=read(p/'confirmation_state.json');assert state['complete'] and state['phase']=='local_confirmation_complete'
limited=a.cohort=='limited';tag='stream_source_shapes_gpu3_'+a.cohort+'_';tasks=[]
settings=([f'g{bw}_l{lat}' for bw in [100,50] for lat in [0,2000,5000,10000,20000]] if limited else ['native_hbm','pcie'])
for i,(mtp,requests) in enumerate([(8,64),(8,128)]):
 for name in ([v,'base'] if i==0 else ['base',v]):
  tasks.append(dict(variant=name,kind='lru',mtp=mtp,requests=requests,settings=settings,fetch_ctas=24,fractions=None))
if not limited:
 for mtp,requests in [(1,1),(8,28),(8,32),(1,256)]:
  for name in ['base',v]:
   tasks.append(dict(variant=name,kind='lru',mtp=mtp,requests=requests,settings=settings,fetch_ctas=16,fractions=None))
 for name in [v,'base']:
  tasks.append(dict(variant=name,kind='nocache',mtp=8,requests=64,settings=settings,fetch_ctas=24,fractions=[0.,.1,.3]))
plan=dict(cohort=a.cohort,hardware='GPU3A100PCIE',base_sha256=sha(p/'base.cuh'),candidate_sha256=sha(p/(v+'.cuh')),positions=1024,passes=3,tasks=tasks,
 expected_cells=sum(len(x['settings'])*(len(x['fractions']) if x['fractions'] else 1) for x in tasks),
 limitation='MTP1R512 independentHBM exceeds GPU3 memory and stays on pending SXM gate. PhysicalNVLink requires GPU0/1 and is not simulated by GPU3. All unmodified36EMU/72normal kernels exact. No new source or parameter scan.')
planfile=p/(tag+'plan.json')
if planfile.exists():assert read(planfile)==plan
else:planfile.write_text(json.dumps(plan,indent=2)+'\n')
if a.plan_only:
 print('PLAN_ONLY',tag,plan['expected_cells'],flush=True)
 raise SystemExit(0)
statefile=p/(tag+'state.json');state=read(statefile) if statefile.exists() else []
for task in tasks:
 remaining=[]
 for setting in task['settings']:
  name=f"{tag}{task['variant']}_staged_f{task['fetch_ctas']}_i64_soff_{task['kind']}_{setting}_mtp{task['mtp']}_b{task['requests']}.json"
  f=p/'gpu3/raw'/name
  if f.exists() and read(f).get('complete'):
   d=read(f);cfg=d['configuration'];assert cfg['passes']==3 and cfg['positions']==1024
   assert d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==sha(p/(task['variant']+'.cuh'))
  else:remaining.append(setting)
 if not remaining:print('SKIP_COMPLETED',task,flush=True);continue
 while not snapshot()['available']:
  print('WAITING_GPU',snapshot(),flush=True);time.sleep(10)
 name=task['variant'];suffix='' if limited else '_normal'
 workspace=(p.parent/'tune_cached_pe_20261101'/('jit_cached_pe' if limited else 'jit_cached_pe_normal')) if name=='base' else p/('jit_'+name+suffix)
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=name,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(workspace),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU' if limited else '')
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),'--kind',task['kind'],'--mtp',str(task['mtp']),'--requests',str(task['requests']),'--settings',*remaining,'--passes','3','--positions','1024','--fetch-ctas',str(task['fetch_ctas']),'--host-split','off']
 if task['fractions'] is not None:cmd+=['--fractions',*map(str,task['fractions'])]
 item=dict(task,start=time.strftime('%FT%T%z'),remaining_settings=remaining,command=cmd);state.append(item)
 print('START_SHAPE',task,flush=True)
 with (p/f"{tag}{name}_{task['kind']}_m{task['mtp']}_b{task['requests']}.log").open('a') as log:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;statefile.write_text(json.dumps(state,indent=2)+'\n');item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');statefile.write_text(json.dumps(state,indent=2)+'\n')
 if item['exit_code']:raise SystemExit(item['exit_code'])
 print('DONE_SHAPE',task,flush=True)
print('SHAPES_COMPLETE',a.cohort,flush=True)
