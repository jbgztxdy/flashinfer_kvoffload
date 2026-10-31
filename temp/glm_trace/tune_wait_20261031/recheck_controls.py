"""Sandwich all unchanged native paths with >1% negative matrix differences."""
import json
import os
import subprocess
import time
from pathlib import Path
from resources_gpu3 import snapshot

p=Path(__file__).resolve().parent;root=p.parents[2]
state=[]
for kind,mtp,requests,settings,fractions in [
    ('nocache',8,32,['pcie'],['0.0','0.1','0.3']),
    ('lru',8,28,['native_hbm'],None),
    ('lru',1,1,['native_hbm','pcie'],None),
]:
    for variant,phase in [('base','before'),('cached_ids','candidate'),('base','after')]:
        tag=f'cached_controls_{phase}_'
        while not snapshot()['available']:
            print('WAITING_GPU3',snapshot(),flush=True);time.sleep(10)
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',
            CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
            TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=variant,TUNE_RUN_TAG=tag,
            FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
        cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),
            '--kind',kind,'--mtp',str(mtp),'--requests',str(requests),'--settings',*settings,
            '--passes','5','--positions','1024','--fetch-ctas','16','--host-split','off']
        if fractions:cmd+=['--fractions',*fractions]
        item=dict(variant=variant,phase=phase,kind=kind,mtp=mtp,requests=requests,command=cmd,start=time.strftime('%FT%T%z'))
        state.append(item)
        with (p/f'{tag}{kind}_m{mtp}_b{requests}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=child.pid
            (p/'recheck_controls_state.json').write_text(json.dumps(state,indent=2)+'\n')
            item['exit_code']=child.wait()
        item['end']=time.strftime('%FT%T%z')
        (p/'recheck_controls_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print('CONTROL_COMPLETE',item,flush=True)
        if item['exit_code']:raise SystemExit(item['exit_code'])
