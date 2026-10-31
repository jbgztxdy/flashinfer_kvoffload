"""Exercise the new path early while the full trace sweep continues between jobs."""
import fcntl,json,os,subprocess,time
from pathlib import Path
from resources import snapshot,require_exclusive
p=Path(__file__).resolve().parent;root=p.parents[2]
with (root/'temp/glm_trace/kernel_report_20261029/gpu01.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 while not snapshot()['available']:time.sleep(30)
 before=require_exclusive()
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),FLASHINFER_WORKSPACE_BASE=str(p/'jit_final_normal'))
 env.pop('FLASHINFER_EXTRA_CUDAFLAGS',None)
 with (p/'pytest_targeted.log').open('w') as stream:
  result=subprocess.run(['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','finalpaths','tests/attention/test_mla_host_batch.py','-k','whole_rows'],env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
 (p/'targeted_tests.json').write_text(json.dumps(dict(exit_code=result.returncode,resources_start=before,resources_end=require_exclusive()),indent=2)+'\n')
 raise SystemExit(result.returncode)
