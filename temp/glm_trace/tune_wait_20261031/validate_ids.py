"""Restart-safe, paired full-window regression for the source-ID sharing candidate."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--hardware', choices=['sxm','gpu3'], required=True)
parser.add_argument('--cohort', choices=['limited','native'], required=True)
a = parser.parse_args()
gpu3 = a.hardware=='gpu3'
assert not gpu3 or a.cohort=='native'
from resources import snapshot as sxm_snapshot
from resources_gpu3 import snapshot as gpu3_snapshot
snapshot = gpu3_snapshot if gpu3 else sxm_snapshot
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(p/'fetch_ids.cuh')=='270c48be82e1d15b70c461b00bac3fe7a124bc75f0b85323960d7e372c4edee9'
gate=json.loads((p/'tests_fetch_ids_gpu3.json').read_text())
assert gate['complete'] and gate['exit_code']==0
shapes=[(8,28),(8,32),(8,40),(8,48),(1,256)]
if not gpu3:
    shapes += [(8,64),(8,128),(1,512)]
settings=([f'g{bw}_l{lat}' for bw in [100,50] for lat in [0,2000,5000,10000,20000]]
    if a.cohort=='limited' else (['native_hbm','pcie'] if gpu3 else ['nvlink','pcie']))
tag='ids_matrix_'+a.hardware+'_'+a.cohort+'_'
tasks=[]
for i,(mtp,requests) in enumerate(shapes):
    for kind in ['lru','nocache']:
        order=['base','fetch_ids'] if i%2==0 else ['fetch_ids','base']
        for v in order:
            tasks.append(dict(variant=v,kind=kind,mtp=mtp,requests=requests,settings=settings,
                fetch_ctas=24 if mtp*requests>=432 else 16,passes=3,
                fractions=[0.,.1,.3] if kind=='nocache' else None))
if a.cohort=='native':
    for mtp,requests in [(1,1),(8,1)]:
        for v in ['base','fetch_ids']:
            tasks.append(dict(variant=v,kind='lru',mtp=mtp,requests=requests,
                settings=settings,fetch_ctas=16,passes=3,fractions=None))
plan=p/(tag+'plan.json')
manifest=dict(hardware=a.hardware,cohort=a.cohort,tag=tag,
    base_sha256=sha(p/'base.cuh'),candidate_sha256=sha(p/'fetch_ids.cuh'),
    positions=1024,full_lru_window=True,no_lru_snapshots=[0,256,512,768],tasks=tasks,
    expected_cells=sum(len(x['settings'])*(len(x['fractions']) if x['fractions'] else 1) for x in tasks))
if plan.exists():
    assert json.loads(plan.read_text())==manifest,'Do not overwrite a queued plan'
else:
    plan.write_text(json.dumps(manifest,indent=2)+'\n')
statepath=p/(tag+'state.json')
state=json.loads(statepath.read_text()) if statepath.exists() else []
rawdir=p/('gpu3/raw' if gpu3 else 'raw')
for task in tasks:
    remaining=[]
    for setting in task['settings']:
        name=f"{tag}{task['variant']}_staged_f{task['fetch_ctas']}_i64_soff_{task['kind']}_{setting}_mtp{task['mtp']}_b{task['requests']}.json"
        f=rawdir/name
        if f.exists() and json.loads(f.read_text()).get('complete'):
            d=json.loads(f.read_text());cfg=d['configuration']
            assert cfg['passes']==3 and cfg['positions']==1024 and cfg['fetch_ctas']==task['fetch_ctas']
            assert d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==sha(p/(task['variant']+'.cuh'))
            continue
        remaining.append(setting)
    if not remaining:
        print('SKIP_COMPLETED',task,flush=True)
        continue
    while not snapshot()['available']:
        print('WAITING_GPU',snapshot(),flush=True)
        time.sleep(10)
    native=a.cohort=='native'
    suffix='_normal' if native else ''
    v=task['variant']
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3' if gpu3 else '0,1',
        CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
        TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=v,TUNE_RUN_TAG=tag,
        FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+suffix)),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
    cmd=['numactl','-N1' if gpu3 else '-N0','-m1' if gpu3 else '-m0',
        '/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/('bench_gpu3_native.py' if gpu3 else 'bench.py')),
        '--kind',task['kind'],'--mtp',str(task['mtp']),'--requests',str(task['requests']),
        '--settings',*remaining,'--passes','3','--positions','1024','--fetch-ctas',str(task['fetch_ctas']),
        '--host-split','off']
    if task['fractions'] is not None:
        cmd += ['--fractions',*map(str,task['fractions'])]
    item=dict(task,start=time.strftime('%FT%T%z'),remaining_settings=remaining,command=cmd)
    state.append(item)
    logpath=p/f"{tag}{v}_{task['kind']}_m{task['mtp']}_b{task['requests']}.log"
    print('START_IDS_REGRESSION',task,flush=True)
    with logpath.open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid
        statepath.write_text(json.dumps(state,indent=2)+'\n')
        item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z')
    statepath.write_text(json.dumps(state,indent=2)+'\n')
    if item['exit_code']:
        raise SystemExit(item['exit_code'])
    print('DONE_IDS_REGRESSION',task,flush=True)
print('IDS_REGRESSION_COMPLETE',a.hardware,a.cohort,flush=True)
