"""Full-window medium dispatch comparisons on GPU3; never mix with SXM/NVLink data."""
import json,os,subprocess,time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
settings=[f'g{bw}_l{lat}' for bw in (100,50) for lat in (0,2000,5000,10000,20000)]+['pcie']
base_env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4')
tasks=[]
# A forward/reverse five-pass confirmation at the screened shape precedes broadening.
for v in ('threshold2','base'):
    tasks.append(dict(variant=v,kind='lru',mtp=8,requests=32,settings=['g100_l0','g100_l20000','g50_l20000'],passes=5,tag='gpu3_confirmfull_'))
# Every affected shape gets a native full-trace reference on this physical GPU.
for m,b in ((8,28),(8,40),(8,48),(1,256)):
    tasks.append(dict(variant='defer_cache',kind='floor',mtp=m,requests=b,settings=['g100_l0'],passes=3,tag='gpu3_floor_'))
for m,b in ((8,28),(8,32),(8,40),(8,48),(1,256)):
    for v in ('base','threshold2'):
        tasks.append(dict(variant=v,kind='lru',mtp=m,requests=b,settings=settings,passes=3,tag='gpu3_full_'))
# Explicit unaffected small/large dispatch controls (same code, same parameters).
for m,b in ((1,1),(8,1),(8,64)):
    tasks.append(dict(variant='defer_cache',kind='floor',mtp=m,requests=b,settings=['g100_l0'],passes=3,tag='gpu3_floor_'))
    for v in ('base','threshold2'):
        tasks.append(dict(variant=v,kind='lru',mtp=m,requests=b,settings=['g100_l20000','g50_l20000','pcie'],passes=3,tag='gpu3_controls_'))
# The native snapshots match each no-LRU residency fraction, on this same GPU.
for m,b in ((8,32),(1,256)):
    tasks.append(dict(variant='defer_cache',kind='snapshots',mtp=m,requests=b,settings=['g100_l0'],passes=3,tag='gpu3_snapshots_',fractions=[0.,.1,.3]))
    for v in ('threshold2','base'):
        tasks.append(dict(variant=v,kind='nocache',mtp=m,requests=b,settings=settings,passes=3,tag='gpu3_nocache_',fractions=[0.,.1,.3]))
state=[]
for task in tasks:
    while not snapshot()['available']:
        print('WAITING_GPU3',snapshot(),flush=True);time.sleep(30)
    v=task['variant'];native=task['kind'] in ('floor','snapshots')
    env=dict(base_env,TUNE_VARIANT=v,TUNE_RUN_TAG=task['tag'],FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+('_normal' if native else ''))),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
    cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_numa1_v2.py'),'--kind',task['kind'],'--mtp',str(task['mtp']),'--requests',str(task['requests']),'--settings',*task['settings'],'--passes',str(task['passes']),'--positions','1024','--fetch-ctas','16','--host-split','off']
    if 'fractions' in task:cmd+=['--fractions',*map(str,task['fractions'])]
    entry=dict(task,start=time.strftime('%FT%T%z'),command=cmd)
    state.append(entry);print('START_GPU3_VALIDATION',task,flush=True)
    with (p/f"{task['tag']}{v}_{task['kind']}_m{task['mtp']}_b{task['requests']}.log").open('a') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid;(p/'gpu3_validation_numa1_v2_state.json').write_text(json.dumps(state,indent=2)+'\n')
        entry['exit_code']=child.wait()
    entry['end']=time.strftime('%FT%T%z');(p/'gpu3_validation_numa1_v2_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if entry['exit_code']:raise SystemExit(entry['exit_code'])
    print('DONE_GPU3_VALIDATION',task,flush=True)
print('GPU3_VALIDATION_COMPLETE',flush=True)
