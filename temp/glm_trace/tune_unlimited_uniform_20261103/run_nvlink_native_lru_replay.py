import os,time,json,subprocess
from pathlib import Path
from resources_nvlink import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
state={'actions':[],'complete':False}
def save():(p/'native_lru_replay_state.json').write_text(json.dumps(state,indent=2))
for v in ['native_lru_replay']:
 stable=None
 for i in range(40):
  s=snapshot()
  if s['available']:
   stable=stable or time.monotonic()
   if time.monotonic()-stable>=10:break
  else:stable=None
  time.sleep(2)
 else:raise RuntimeError(s)
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(p)+':'+str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT="retained",FLASHINFER_WORKSPACE_BASE=str(p/('jit_retained')) if v!='base' else str(p.parent/'emu_fix_20261103/jit_v2_normal'),FLASHINFER_EXTRA_CUDAFLAGS='')
 cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/('bench_nvlink_'+v+'.py')),'--kind','lru','--mtp','8','--requests','128','--positions','1024','--passes','3','--settings','native_hbm','--fetch-ctas','24','--host-split','off']
 a={'variant':v,'start':time.strftime('%FT%T%z'),'command':cmd,'preflight':s};state['actions'].append(a);save();print('START',v,flush=True)
 with (p/('nvlink_'+v+'_retry.log')).open('x') as f:
  telemetry=(p/'native_lru_replay_clocks.csv').open('x')
  monitor=subprocess.Popen(['nvidia-smi','--id=0','--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv','-lms','50'],stdout=telemetry)
  child=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT);a['pid']=child.pid;save();a['exit_code']=child.wait();monitor.terminate();monitor.wait();telemetry.close()
 save();print('END',v,a['exit_code'],flush=True)
 if a['exit_code']:raise RuntimeError(v)
state['complete']=True;save()
