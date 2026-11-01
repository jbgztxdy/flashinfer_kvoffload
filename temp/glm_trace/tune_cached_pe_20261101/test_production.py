"""Test the installed header with a fresh normal JIT, independently of candidate trees."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from resources_gpu3 import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
expected=sha(p/'cached_pe.cuh')
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==expected
log=p/'compile_production.log'
while 'COMPILED_PRODUCTION_NORMAL' not in log.read_text():
    if 'Traceback' in log.read_text():raise SystemExit('Production CPU compile failed; inspect preserved log')
    print('WAITING_PRODUCTION_CPU_BUILD',flush=True);time.sleep(5)
libraries=list((p/'jit_production_cached_pe_normal').rglob('*.so'));assert len(libraries)==4
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3',snapshot(),flush=True);time.sleep(10)
record=dict(start=time.strftime('%FT%T%z'),production_header_sha256=expected,
    math_header_sha256=sha(root/'include/flashinfer/attention/mla.cuh'),
    test_source_sha256=sha(root/'tests/attention/test_mla_host_batch.py'),
    stress_source_sha256=sha(p/'test_cofetch_progress.py'),expected_tests=111,resources_start=snapshot(),
    fresh_workspace=str(p/'jit_production_cached_pe_normal'),include_path=str(root/'include'),normal_build=True,
    libraries=[dict(path=str(f),sha256=sha(f)) for f in libraries],complete=False)
out=p/'tests_production_gpu3.json';out.write_text(json.dumps(record,indent=2)+'\n')
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',
    FLASHINFER_CUDA_ARCH_LIST='8.0',FLASHINFER_EXTRA_CUDAFLAGS='',MAX_JOBS='4',
    FLASHINFER_WORKSPACE_BASE=str(p/'jit_production_cached_pe_normal'),TMPDIR=str(root/'temp'),
    PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root))
env.pop('FLASHINFER_FAR_EMU',None)
command=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q',
    '-p','no:cacheprovider','-p','jitpaths','-p','productionpaths','tests/attention/test_mla_host_batch.py',str(p/'test_cofetch_progress.py')]
with (p/'pytest_production_gpu3.log').open('w') as f:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
    record['pid']=child.pid;record['command']=command;out.write_text(json.dumps(record,indent=2)+'\n')
    record['exit_code']=child.wait()
end=snapshot()
while end['foreign'] and all(r['pid']==child.pid for r in end['foreign']):
    time.sleep(.2);end=snapshot()
record.update(end=time.strftime('%FT%T%z'),resources_end=end,log_sha256=sha(p/'pytest_production_gpu3.log'))
record['complete']=record['exit_code']==0 and not end['foreign'] and sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==expected
out.write_text(json.dumps(record,indent=2)+'\n');print('PRODUCTION_TEST_COMPLETE',record['complete'],record['exit_code'])
raise SystemExit(0 if record['complete'] else 1)
