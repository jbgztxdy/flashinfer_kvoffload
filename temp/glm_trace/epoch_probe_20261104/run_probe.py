import sys,os,time,json,subprocess,fcntl,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=root/'temp/glm_trace/tune_unlimited_uniform_20261103';sys.path.insert(0,str(old));from resources_gpu3 import snapshot
lock=(old.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stable=None
for i in range(40):
 s=snapshot()
 if s['available']:
  stable=stable or time.monotonic()
  if time.monotonic()-stable>=10:break
 else:stable=None
 time.sleep(2)
else:raise RuntimeError(s)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID');before=snapshot()
with (p/'probe_raw.json').open('x') as f:
 r=subprocess.run(['numactl','-N1','-m1',str(p/'probe')],env=env,stdout=f,stderr=subprocess.PIPE,text=True)
after=snapshot();(p/'probe_state.json').write_text(json.dumps(dict(before=before,after=after,exit_code=r.returncode,stderr=r.stderr,source_sha256=hashlib.sha256((p/'probe.cu').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((p/'probe').read_bytes()).hexdigest()),indent=2));assert r.returncode==0 and not after['foreign']
