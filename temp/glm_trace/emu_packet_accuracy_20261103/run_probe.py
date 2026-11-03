import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sys.path.insert(0,str(p))
from resources_gpu3 import snapshot,require_exclusive
r=dict(start=time.strftime('%FT%T%z'),complete=False,preflight=[])
assert not(p/'probe_result.json').exists()
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stable=None;deadline=time.monotonic()+120
while True:
 s=snapshot();r['preflight'].append(s);(p/'probe_state.json').write_text(json.dumps(r,indent=2)+'\n')
 if s['available']:
  if stable is None:stable=time.monotonic()
  if time.monotonic()-stable>=10:break
 else:stable=None
 if time.monotonic()>deadline:
  r['status']='preflight_timeout_before_CUDA';r['end']=time.strftime('%FT%T%z');(p/'probe_state.json').write_text(json.dumps(r,indent=2)+'\n');raise RuntimeError('GPU3 busy')
 time.sleep(2)
r['resources_start']=require_exclusive()
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3')
cmd=['numactl','-N1','-m1',str(p/'probe')];r['command']=cmd
with(p/'probe_stdout.log').open('w') as out,(p/'probe_stderr.log').open('w') as err:
 child=subprocess.Popen(cmd,env=env,cwd=root,stdout=out,stderr=err);r['pid']=child.pid;(p/'probe_state.json').write_text(json.dumps(r,indent=2)+'\n');r['exit_code']=child.wait()
r['end']=time.strftime('%FT%T%z');r['resources_end']=snapshot()
(p/'probe_state.json').write_text(json.dumps(r,indent=2)+'\n')
assert r['exit_code']==0,(p/'probe_stderr.log').read_text()
d=json.loads((p/'probe_stdout.log').read_text());d['provenance']=r;d['source_sha256']=hashlib.sha256((p/'probe.cu').read_bytes()).hexdigest();d['binary_sha256']=hashlib.sha256((p/'probe').read_bytes()).hexdigest();d['far_sha256']=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest()
assert not r['resources_start']['foreign'] and not r['resources_end']['foreign'];d['complete']=True
(p/'probe_result.json').write_text(json.dumps(d,indent=2)+'\n');r['complete']=True;(p/'probe_state.json').write_text(json.dumps(r,indent=2)+'\n')
print('DONE_TRANSFER',flush=True)
