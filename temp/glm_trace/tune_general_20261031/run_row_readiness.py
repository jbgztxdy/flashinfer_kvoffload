"""Serial paired readiness screening; complete trace checks are a separate stage."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
base_env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
state=[]
for variant in ('row_ready','base'):
    while not snapshot()['available']:
        print('WAITING',snapshot(),flush=True)
        time.sleep(30)
    env=dict(base_env,TUNE_VARIANT=variant,TUNE_RUN_TAG='screenrow256_',FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant)))
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','nvlink','--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
    entry=dict(variant=variant,start=time.strftime('%FT%T%z'),command=cmd)
    state.append(entry)
    print('START',variant,entry['start'],flush=True)
    with (p/('screenrow256_'+variant+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid
        (p/'row_readiness_state.json').write_text(json.dumps(state,indent=2)+'\n')
        entry['exit_code']=child.wait()
    entry['end']=time.strftime('%FT%T%z')
    (p/'row_readiness_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if entry['exit_code']:raise SystemExit(entry['exit_code'])
    print('DONE',variant,entry['end'],flush=True)
print('SCREEN_COMPLETE',flush=True)
