"""Two v2 link points plus a fresh normal compute reference. One CPU waiter, no old runner."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources_gpu3 import snapshot
r=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[],preflight_observations=[])
def save():(p/'state.json').write_text(json.dumps(r,indent=2)+'\n')
def wait_free():
 deadline=time.monotonic()+600
 while True:
  obs=snapshot();r['preflight_observations'].append(obs);save()
  if obs['available']:return
  if time.monotonic()>deadline:raise RuntimeError('GPU3 remained occupied; no GPU task started for this action')
  print('WAIT_GPU3',obs['timestamp'],[x['pid'] for x in obs['foreign']],flush=True);time.sleep(10)
def run(label,kind,settings,normal):
 wait_free()
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='current',TUNE_RUN_TAG='v2_256_',FLASHINFER_WORKSPACE_BASE=str(p.parent/'emu_fix_20261103'/('jit_v2_normal' if normal else 'jit_v2')),FLASHINFER_EXTRA_CUDAFLAGS='' if normal else '-DFLASHINFER_FAR_EMU')
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind',kind,'--mtp','8','--requests','128','--settings',*settings,'--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
 action=dict(label=label,start=time.strftime('%FT%T%z'),command=cmd);r['actions'].append(action);save()
 with (p/(label+'.log')).open('w') as log:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);action['pid']=child.pid;save();print('START',label,child.pid,flush=True);action['exit_code']=child.wait()
 action['end']=time.strftime('%FT%T%z');save();print('END',label,action['exit_code'],flush=True)
 assert action['exit_code']==0,label
m=json.loads((p/'manifest.json').read_text())
for rel,digest in m['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
for lib in m['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
save()
run('link100', 'lru',['g100_l0','g100_l20000'],False)
run('normal_compute','floor',['floor'],True)
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
