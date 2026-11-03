from pathlib import Path
import os,json,time,hashlib,subprocess,fcntl
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pre=[]
for _ in range(6):
 s=snapshot();pre.append(s);assert s['available'],s;time.sleep(2)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3')
cmd=['numactl','-N1','-m1',str(p/'demand_consume')]
with (p/'demand_consume_run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
record=dict(command=cmd,preflight=pre,returncode=r.returncode,resources_end=snapshot(),source_sha256=hashlib.sha256((p/'demand_consume.cu').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((p/'demand_consume').read_bytes()).hexdigest(),scope='payload protocol only; no attention math or performance claim')
(p/'demand_consume_run.json').write_text(json.dumps(record,indent=2)+'\n');assert r.returncode==0
print((p/'demand_consume_run.log').read_text())
