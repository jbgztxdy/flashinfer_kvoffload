"""Wait for the performance matrix, then run the full suite with GPU01 exclusive."""
import fcntl,json,os,subprocess,time
from pathlib import Path
from resources import snapshot,require_exclusive
p=Path(__file__).resolve().parent
root=p.parents[2]
previous=json.loads((p/'signal_validation.json').read_text())['child_pid']
while True:
 try:os.kill(previous,0)
 except ProcessLookupError:break
 time.sleep(30)
while not snapshot()['available']:time.sleep(30)
with (root/'temp/glm_trace/kernel_report_20261029/gpu01.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 before=require_exclusive()
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),FLASHINFER_WORKSPACE_BASE=str(p/'jit_final_normal'))
 env.pop('FLASHINFER_EXTRA_CUDAFLAGS',None)
 with (p/'pytest_final.log').open('w') as stream:
  result=subprocess.run(['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','finalpaths','tests/attention/test_mla_host_batch.py'],env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
 after=require_exclusive()
 (p/'tests.json').write_text(json.dumps(dict(exit_code=result.returncode,resources_start=before,resources_end=after),indent=2)+'\n')
 raise SystemExit(result.returncode)
