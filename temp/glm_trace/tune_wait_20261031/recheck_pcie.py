"""Recheck the actual native PCIe no-LRU regression, bounded by two base runs."""
import json
import os
import subprocess
import time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
while True:
    gate=p/'tests_cached_ids_gpu3.json'
    t=json.loads(gate.read_text()) if gate.exists() else {}
    if t.get('complete'):break
    if 'exit_code' in t:raise SystemExit('Cached-only correctness gate failed')
    print('WAITING_CACHED_TESTS',flush=True);time.sleep(20)
state=[]
for v,tag in [('base','ids_pcie_recheck_'),('fetch_ids','ids_pcie_recheck_'),('cached_ids','ids_pcie_recheck_'),('base','ids_pcie_recheck_after_')]:
    while not snapshot()['available']:
        print('WAITING_GPU3',snapshot(),flush=True);time.sleep(10)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',
        FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',
        TUNE_VARIANT=v,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
    cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),
        '--kind','nocache','--mtp','8','--requests','32','--settings','pcie','--passes','5','--positions','1024',
        '--fetch-ctas','16','--host-split','off','--fractions','0.0','0.1','0.3']
    item=dict(variant=v,tag=tag,start=time.strftime('%FT%T%z'),command=cmd);state.append(item)
    with (p/(tag+v+'.log')).open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid;(p/'pcie_recheck_state.json').write_text(json.dumps(state,indent=2)+'\n')
        item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');(p/'pcie_recheck_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('RECHECK_DONE',v,item['exit_code'],flush=True)
    if item['exit_code']:raise SystemExit(item['exit_code'])
