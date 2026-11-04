"""Standalone GPU3 same-device HBM probe. Compile without allocating GPU memory."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p=Path(__file__).resolve().parent
root=p.parents[2]
sys.path.insert(0,str(p.parent/'tune_unlimited_uniform_20261103'))
from resources_gpu3 import snapshot

a=argparse.ArgumentParser()
a.add_argument('--compile-only',action='store_true')
a.add_argument('--wait-seconds',type=int,default=60)
args=a.parse_args()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
binary=p/'probe'
if not binary.exists() or (p/'probe.cu').stat().st_mtime>binary.stat().st_mtime:
 subprocess.run(['/usr/local/cuda-13.1/bin/nvcc','-O3','-std=c++17','-arch=sm_80',str(p/'probe.cu'),'-o',str(binary)],check=True,cwd=root,env=dict(os.environ,TMPDIR=str(root/'temp')))
if args.compile_only:
 print('COMPILED',sha(binary));sys.exit(0)
stamp=time.strftime('%Y%m%d_%H%M%S')
record=dict(source_sha256=sha(p/'probe.cu'),binary_sha256=sha(binary),preflight=[],complete=False)
out=p/('result_'+stamp+'.json')
def save():out.write_text(json.dumps(record,indent=2)+'\n')
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
deadline=time.monotonic()+args.wait_seconds
stable=None
while True:
 s=snapshot();record['preflight'].append(s);save()
 if s['available']:
  if stable is None:stable=time.monotonic()
  if time.monotonic()-stable>=10:break
 else:stable=None
 if time.monotonic()>=deadline:
  record['not_run']='GPU3 did not remain exclusively idle for 10 seconds';save();print(record['not_run'],out);sys.exit(2)
 time.sleep(2)
fields='index,uuid,name,clocks.sm,clocks.mem,power.draw,temperature.gpu'
record['hardware_start']=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu='+fields,'--format=csv,noheader'],text=True).strip()
cmd=['numactl','-N1','-m1',str(binary)]
record['command']=cmd
proc=subprocess.run(cmd,cwd=root,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3'),text=True,capture_output=True,timeout=120)
(p/('raw_'+stamp+'.jsonl')).write_text(proc.stdout)
(p/('stderr_'+stamp+'.log')).write_text(proc.stderr)
record['exit_code']=proc.returncode
record['resources_end']=snapshot()
record['hardware_end']=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu='+fields,'--format=csv,noheader'],text=True).strip()
record['measurements']=[json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
save()  # Preserve failures/interference before any validation assertion.
assert proc.returncode==0,proc.stderr
assert not record['resources_end']['foreign'],record['resources_end']
assert record['measurements'][-1].get('payload_checks')
for row in record['measurements']:
 if row['kind']=='bandwidth':
  row['median_us']=statistics.median(row['us'])
  row['payload_GBps']=row['payload_bytes']/row['median_us']/1000
  row['read_plus_write_GBps']=2*row['payload_GBps']
 if row['kind']=='dependent_load_latency':row['median_ns']=statistics.median(row['ns_per_hop'])
record['complete']=True;save()
print(json.dumps(record['measurements'],indent=2));print(out)
