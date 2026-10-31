"""Serialize candidate experiments and wait for GPUs without disturbing other jobs."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
tasks=[('fetch128',16,'on'),('fetch128',24,'on'),('fetch128',32,'on')]
previous=json.loads((p/'signal_stage4.json').read_text())['child_pid']
while True:
    try:os.kill(previous,0)
    except ProcessLookupError:break
    time.sleep(30)

for v,f,s in tasks:
    while True:
        state=snapshot()
        if state['available']:break
        (p/'waiting.json').write_text(json.dumps(state,indent=2)+'\n')
        print('WAITING',state['timestamp'],flush=True)
        time.sleep(30)
    current=dict(env,TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v)))
    print('START',v,f,s,time.strftime('%FT%T%z'),flush=True)
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','nvlink','--fetch-ctas',str(f),'--host-split',s]
    with (p/f'{v}_f{f}_s{s}.log').open('w') as stream:r=subprocess.run(cmd,env=current,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit(r.returncode)
    print('DONE',v,f,s,flush=True)
print('CANDIDATES_COMPLETE',flush=True)
