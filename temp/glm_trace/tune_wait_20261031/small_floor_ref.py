"""Replace no original data: add the missing-provenance small normal SXM reference."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
while not snapshot()['available']:
    print('WAITING_GPU',snapshot(),flush=True);time.sleep(10)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',
    CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
    TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT='base',TUNE_RUN_TAG='cached_small_ref_',
    FLASHINFER_WORKSPACE_BASE=str(p/'jit_base_normal'),FLASHINFER_EXTRA_CUDAFLAGS='')
command=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),
    '--kind','floor','--mtp','1','--requests','1','--positions','1024','--passes','3',
    '--fetch-ctas','16','--host-split','off']
state=dict(command=command,start=time.strftime('%FT%T%z'))
with (p/'small_floor_ref_bench.log').open('a') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['pid']=child.pid;(p/'small_floor_ref_state.json').write_text(json.dumps(state,indent=2)+'\n')
    state['exit_code']=child.wait()
state['end']=time.strftime('%FT%T%z');(p/'small_floor_ref_state.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(state['exit_code'])
