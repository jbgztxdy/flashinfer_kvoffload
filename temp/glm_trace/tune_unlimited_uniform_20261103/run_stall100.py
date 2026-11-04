import os,time,json,subprocess
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
state={'actions':[],'complete':False}
def save():(p/'stall100_state.json').write_text(json.dumps(state,indent=2))
for phase,variant,flags in [('stall','stall100','-DFLASHINFER_FAR_EMU')]:
 stable=None
 for i in range(40):
  s=snapshot()
  if s['available']:
   stable=stable or time.monotonic()
   if time.monotonic()-stable>=10:break
  else:stable=None
  time.sleep(2)
 else:raise RuntimeError(s)
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(p)+':'+str(root),TMPDIR=str(root/'temp'),FLASHINFER_STALL_PS_PER_BYTE='2000',MAX_JOBS='2',TUNE_VARIANT=variant,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant)),FLASHINFER_EXTRA_CUDAFLAGS=flags)
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_stall100.py'),'--kind','lru','--mtp','8','--requests','128','--positions','1024','--passes','1','--settings','native_hbm','--fetch-ctas','24','--host-split','off']
 a={'phase':phase,'start':time.strftime('%FT%T%z'),'command':cmd,'preflight':s};state['actions'].append(a);save();print('START',phase,flush=True)
 with (p/('stall100_'+phase+'.log')).open('x') as f:
  child=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT);a['pid']=child.pid;save();a['exit_code']=child.wait()
 save();print('END',phase,a['exit_code'],flush=True)
 if a['exit_code']:raise RuntimeError(phase)
state['complete']=True;save()
