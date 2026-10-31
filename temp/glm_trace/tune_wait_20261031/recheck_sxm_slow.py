"""Bracket the remaining slow-path concerns after the complete cached-ID matrix."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from resources import snapshot

p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(p/'cached_ids.cuh')=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
tests=json.loads((p/'tests_cached_ids_gpu3.json').read_text())
assert tests['complete'] and tests['exit_code']==0
statepath=p/'sxm_slow_recheck_state.json'
state=json.loads(statepath.read_text()) if statepath.exists() else []
tasks=[dict(kind='lru',mtp=8,requests=128,fetch=24,native=False,
            settings=['g100_l0','g100_l2000','g100_l5000','g100_l10000'],fractions=None),
       dict(kind='nocache',mtp=8,requests=40,fetch=16,native=True,
            settings=['pcie'],fractions=['0.3'])]
for task in tasks:
    for variant,phase in [('base','before'),('cached_ids','candidate'),('base','after')]:
        tag='cached_sxm_slow_'+phase+'_'
        while not snapshot()['available']:
            print('WAITING_GPU',snapshot(),flush=True);time.sleep(10)
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',
            CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
            TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=variant,TUNE_RUN_TAG=tag,
            FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant+('_normal' if task['native'] else ''))),
            FLASHINFER_EXTRA_CUDAFLAGS='' if task['native'] else '-DFLASHINFER_FAR_EMU')
        cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),
             '--kind',task['kind'],'--mtp',str(task['mtp']),'--requests',str(task['requests']),
             '--settings',*task['settings'],'--positions','1024','--passes','5',
             '--fetch-ctas',str(task['fetch']),'--host-split','off']
        if task['fractions']:cmd+=['--fractions',*task['fractions']]
        item=dict(task,variant=variant,phase=phase,command=cmd,start=time.strftime('%FT%T%z'))
        state.append(item)
        with (p/f'{tag}{task["kind"]}_{variant}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=child.pid;statepath.write_text(json.dumps(state,indent=2)+'\n')
            item['exit_code']=child.wait()
        item['end']=time.strftime('%FT%T%z');statepath.write_text(json.dumps(state,indent=2)+'\n')
        print('SLOW_RECHECK_DONE',item['kind'],variant,phase,item['exit_code'],flush=True)
        if item['exit_code']:raise SystemExit(item['exit_code'])
