"""Final trace sweep, focused no-LRU matrix, and unchanged-link regression samples."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
previous=json.loads((p/'signal_stage5.json').read_text())['child_pid']
while True:
    try:os.kill(previous,0)
    except ProcessLookupError:break
    time.sleep(30)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
meta=json.loads((p/'experiment.json').read_text())
regression={(1,1),(1,4),(1,16),(1,128),(1,512),(8,1),(8,4),(8,16),(8,32),(8,64),(8,128)}
settings=['g100_l0','g100_l2000','g100_l5000','g100_l10000','g100_l20000']
tasks=[]
# All 39 shapes at all five 100 GB/s latencies.
for mtp,bs in meta['batches'].items():
    for batch in bs:
        tasks.append(('final','lru',int(mtp),batch,settings+(['g50_l20000','pcie','nvlink'] if (int(mtp),batch) in regression else []),True,3))
# Repeat old default on changed large shapes for a same-session control.
for m,b in sorted(regression):
    if m*b>=432:tasks.append(('base','lru',m,b,settings+['g50_l20000','pcie','nvlink'],False,3))
# Four representative shapes, all seven remote fractions, with and without the optimization.
for m,b in ((1,1),(1,512),(8,16),(8,128)):
    for v in ('base','final'):
        tasks.append((v,'nocache',m,b,['g100_l0','g100_l20000','g50_l20000'],v=='final',3))
for v,kind,m,b,ss,auto,passes in tasks:
    while True:
        state=snapshot()
        if state['available']:break
        (p/'waiting_validation.json').write_text(json.dumps(state,indent=2)+'\n')
        print('WAITING',state['timestamp'],flush=True);time.sleep(30)
    print('START',v,kind,m,b,ss,time.strftime('%FT%T%z'),flush=True)
    current=dict(env,TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v)))
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind',kind,'--mtp',str(m),'--requests',str(b),'--settings',*ss,'--passes',str(passes)]
    if auto:cmd.append('--auto')
    with (p/f'validate_{v}_{kind}_m{m}_b{b}.log').open('w') as stream:r=subprocess.run(cmd,env=current,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit(r.returncode)
    print('DONE',v,kind,m,b,flush=True)
print('FINAL_VALIDATION_COMPLETE',flush=True)
