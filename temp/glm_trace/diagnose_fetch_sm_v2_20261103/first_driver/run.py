"""One exclusive GPU3 diagnostic sequence. No production edits or old runners."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
r=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[],preflight=[])
def save():(p/'run_state.json').write_text(json.dumps(r,indent=2)+'\n')
def idle():
 start=time.monotonic();stable=None
 while time.monotonic()-start<600:
  s=snapshot();r['preflight'].append(s);save()
  if s['available']:
   if stable is None:stable=time.monotonic()
   if time.monotonic()-stable>=10:return
  else:stable=None
  time.sleep(2 if s['available'] else 10)
 raise RuntimeError('GPU3 unavailable; no launch')
m=json.loads((p/'manifest.json').read_text())
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['production_sha256']
for v,settings,passes in [('trace',['g100_l0','g100_l20000'],1),('capacity',['g100_l0'],3)]:
 idle()
 assert hashlib.sha256((p/(v+'.cuh')).read_bytes()).hexdigest()==m['variants'][v]
 assert (p/(v+'.cuh')).read_bytes()==(p/('include_'+v)/'flashinfer/attention/mla_host_batch.cuh').read_bytes()
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v)),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes',str(passes),'--settings',*settings,'--fetch-ctas','24','--host-split','off']
 a=dict(variant=v,command=cmd,start=time.strftime('%FT%T%z'));r['actions'].append(a);save()
 with (p/('run_'+v+'.log')).open('w') as f:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT);a['pid']=child.pid;save();print('START',v,child.pid,flush=True);a['exit_code']=child.wait()
 a['end']=time.strftime('%FT%T%z');save();print('END',v,a['exit_code'],flush=True)
 assert a['exit_code']==0
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
