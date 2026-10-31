"""Native GPU3 reference and paired medium-batch screen, separate from SXM experiments."""
import json,os,subprocess,time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
base_env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4')
tasks=[('defer_cache','floor',1024,3,'off','gpu3_floor_',False),('base','lru',256,3,'off','gpu3_medium_',True),('threshold2','lru',256,3,'off','gpu3_medium_',True),('base','lru',256,3,'on','gpu3_medium_default_',True)]
state=[]
for v,kind,positions,passes,split,tag,emu in tasks:
    while not snapshot()['available']:
        print('WAITING_GPU3',snapshot(),flush=True);time.sleep(30)
    env=dict(base_env,TUNE_VARIANT=v,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+('' if emu else '_normal'))),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU' if emu else '')
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3.py'),'--kind',kind,'--mtp','8','--requests','32','--settings','g100_l0','g100_l20000','g50_l20000','--passes',str(passes),'--positions',str(positions),'--fetch-ctas','16','--host-split',split]
    entry=dict(variant=v,kind=kind,positions=positions,passes=passes,split=split,command=cmd,start=time.strftime('%FT%T%z'))
    state.append(entry)
    print('START_GPU3',v,kind,split,flush=True)
    with (p/f'{tag}{v}_{kind}_{split}.log').open('w') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid;(p/'gpu3_runner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        entry['exit_code']=child.wait()
    entry['end']=time.strftime('%FT%T%z');(p/'gpu3_runner_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if entry['exit_code']:raise SystemExit(entry['exit_code'])
    print('DONE_GPU3',v,kind,split,flush=True)
print('GPU3_SCREEN_COMPLETE',flush=True)
