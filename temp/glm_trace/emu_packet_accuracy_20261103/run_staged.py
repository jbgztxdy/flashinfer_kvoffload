"""Focused correctness then two fresh paired performance points. No production mutation."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
r=dict(start=time.strftime('%FT%T%z'),complete=False,preflight=[],actions=[])
def save():(p/'run_state.json').write_text(json.dumps(r,indent=2)+'\n')
def idle():
 deadline=time.monotonic()+180;stable=None
 while True:
  s=snapshot();r['preflight'].append(s);save()
  if s['available']:
   if stable is None:stable=time.monotonic()
   if time.monotonic()-stable>=10:return
  else:stable=None
  if time.monotonic()>deadline:raise RuntimeError('GPU3 busy; action not started')
  time.sleep(2 if s['available'] else 10)
def run(label,variant,args,test=False):
 idle();lock=None
 if test:
  lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=variant,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant)) if variant!='base' else str(p.parent/'emu_fix_20261103/jit_v2'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python',*args]
 a=dict(label=label,variant=variant,start=time.strftime('%FT%T%z'),command=cmd,resources_start=snapshot());r['actions'].append(a);save();print('START',label,flush=True)
 with(p/(label+'.log')).open('w') as f:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT);a['pid']=child.pid;save()
  try:a['exit_code']=child.wait(timeout=120)
  except subprocess.TimeoutExpired:
   child.terminate();a['timeout']=True
   try:a['exit_code']=child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();a['exit_code']=child.wait()
 a['end']=time.strftime('%FT%T%z');a['resources_end']=snapshot();save()
 if lock:lock.close()
 print('END',label,a['exit_code'],flush=True);assert a['exit_code']==0 and not a['resources_end']['foreign'],label
save()
for v in ['staged','uniform']:
 run('correctness_'+v,v,['-m','pytest','-q','-s','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths',str(p/('test_'+v+'.py'))],True)
for setting,order in [('g100_l0',['base','staged','uniform']),('g100_l20000',['uniform','staged','base'])]:
 for v in order:
  run(v+'_'+setting,v,['-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes','3','--settings',setting,'--fetch-ctas','24','--host-split','off'])
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
