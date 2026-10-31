"""Use free GPU3 for two large shapes while SXM resources are occupied.

R512/MTP1 same-device HBM would exceed40GB, so it remains in the two-GPU SXM
plan. These GPU3 shapes fit and have existing same-hardware full local floors.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
gate=json.loads((p/'tests_cached_ids_gpu3.json').read_text());assert gate['complete'] and gate['exit_code']==0
tag='cached_ids_matrix_gpu3_large_'
tasks=[]
for b in [64,128]:
    for kind in ['lru','nocache']:
        for v in (['cached_ids','base'] if b==64 else ['base','cached_ids']):
            tasks.append(dict(variant=v,kind=kind,mtp=8,requests=b,settings=['native_hbm','pcie'],fetch_ctas=24,
                passes=3,fractions=[0.,.1,.3] if kind=='nocache' else None))
manifest=dict(hardware='gpu3',cohort='large_native',tag=tag,positions=1024,tasks=tasks,expected_cells=32,
    note='Only fitting R64/128,MTP8 on same-card HBM/NUMA1 PCIe. Normal snapshots measured first; full LRU floors from prior GPU3 audit.')
plan=p/(tag+'plan.json')
if plan.exists():assert json.loads(plan.read_text())==manifest
else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
statefile=p/(tag+'state.json');state=json.loads(statefile.read_text()) if statefile.exists() else []
references=[dict(variant='base',kind='snapshots',mtp=8,requests=b,settings=['floor'],fetch_ctas=24,passes=3,fractions=[0.,.1,.3],tag='cached_reference_') for b in [64,128]]
for task in references+[dict(t,tag=tag) for t in tasks]:
    remaining=[]
    for setting in task['settings']:
        f=p/'gpu3/raw'/f"{task['tag']}{task['variant']}_staged_f24_i64_soff_{task['kind']}_{setting}_mtp8_b{task['requests']}.json"
        if f.exists() and json.loads(f.read_text()).get('complete'):
            assert json.loads(f.read_text())['configuration']['passes']==3
            continue
        remaining.append(setting)
    if not remaining:continue
    while not snapshot()['available']:
        print('WAITING_GPU3',snapshot(),flush=True);time.sleep(10)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',
        FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=task['variant'],
        TUNE_RUN_TAG=task['tag'],FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+task['variant']+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
    cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),
        '--kind',task['kind'],'--mtp','8','--requests',str(task['requests']),'--settings',*remaining,
        '--passes','3','--positions','1024','--fetch-ctas','24','--host-split','off']
    if task['fractions'] is not None:cmd+=['--fractions',*map(str,task['fractions'])]
    item=dict(task,start=time.strftime('%FT%T%z'),command=cmd);state.append(item)
    print('START_GPU3_LARGE',task,flush=True)
    with (p/f"{task['tag']}{task['variant']}_{task['kind']}_b{task['requests']}.log").open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid
        statefile.write_text(json.dumps(state,indent=2)+'\n');item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');statefile.write_text(json.dumps(state,indent=2)+'\n')
    if item['exit_code']:raise SystemExit(item['exit_code'])
    print('DONE_GPU3_LARGE',task,flush=True)
print('GPU3_LARGE_COMPLETE',flush=True)
