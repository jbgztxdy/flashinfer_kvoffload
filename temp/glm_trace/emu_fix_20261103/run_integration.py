"""Three focused integration cases, under the shared exclusive GPU3 lock."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources_gpu3 import snapshot
assert all(r['exit_code']==0 for r in json.loads((p/'build_v2.json').read_text()))
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=snapshot();assert before['available'],before
r=dict(start=time.strftime('%FT%T%z'),complete=False,resources_start=before)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='v2',FLASHINFER_WORKSPACE_BASE=str(p/'jit_v2'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths',str(p/'test_v2.py')]
r['command']=cmd;(p/'integration.json').write_text(json.dumps(r,indent=2)+'\n')
with (p/'integration.log').open('w') as f:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT);r['pid']=child.pid;r['exit_code']=child.wait()
r.update(end=time.strftime('%FT%T%z'),resources_end=snapshot(),complete=r['exit_code']==0)
assert not r['resources_end']['foreign'],r['resources_end'];(p/'integration.json').write_text(json.dumps(r,indent=2)+'\n')
print((p/'integration.log').read_text());assert r['complete']
