"""Full-window medium dispatch comparisons on GPU3; never mix with SXM/NVLink data."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
settings=[f'g{bw}_l{lat}' for bw in (100,50) for lat in (0,2000,5000,10000,20000)]+['pcie']
base_env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4')
tasks=[]
for m,b in ((8,28),(8,32),(8,40),(8,48),(1,256)):
    for kind in ('lru','nocache'):
        for v in ('base','balanced_cached_sm80'):
            task=dict(variant=v,kind=kind,mtp=m,requests=b,settings=['g100_l0','g100_l20000','g50_l20000'],passes=3,tag='balanced_sxm_full_')
            if kind=='nocache':task['fractions']=[0.,.1,.3]
            tasks.append(task)
state=[]
for task in tasks:
    while not snapshot()['available']:
        print('WAITING_SXM01',snapshot(),flush=True);time.sleep(30)
    v=task['variant'];native=False
    env=dict(base_env,TUNE_VARIANT=v,TUNE_RUN_TAG=task['tag'],FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+('_normal' if native else ''))),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind',task['kind'],'--mtp',str(task['mtp']),'--requests',str(task['requests']),'--settings',*task['settings'],'--passes',str(task['passes']),'--positions','1024','--fetch-ctas','16','--host-split','off']
    if 'fractions' in task:cmd+=['--fractions',*map(str,task['fractions'])]
    entry=dict(task,start=time.strftime('%FT%T%z'),command=cmd)
    state.append(entry);print('START_SXM_VALIDATION',task,flush=True)
    with (p/f"{task['tag']}{v}_{task['kind']}_m{task['mtp']}_b{task['requests']}.log").open('a') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid;(p/'balanced_sxm_full_state.json').write_text(json.dumps(state,indent=2)+'\n')
        entry['exit_code']=child.wait()
    entry['end']=time.strftime('%FT%T%z');(p/'balanced_sxm_full_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if entry['exit_code']:raise SystemExit(entry['exit_code'])
    print('DONE_SXM_VALIDATION',task,flush=True)
print('SXM_VALIDATION_COMPLETE',flush=True)
