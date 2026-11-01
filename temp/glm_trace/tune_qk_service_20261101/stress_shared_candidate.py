"""Run the additional long-queue/cache-payload gate after the normal 105 tests."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True, choices=['qks_shared','retained_cached_pe'])
a = parser.parse_args()
from resources_gpu3 import snapshot, require_exclusive
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
gate = p/('tests_'+a.variant+'_gpu3.json')
while True:
    state = json.loads(gate.read_text()) if gate.exists() else {}
    if state.get('complete'):
        assert state['exit_code'] == 0
        break
    if state.get('exit_code') is not None:
        raise SystemExit('Original correctness gate failed')
    print('WAITING_NORMAL_CORRECTNESS', flush=True)
    time.sleep(15)
lock = (p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3', snapshot(), flush=True)
    time.sleep(15)
record = dict(variant=a.variant, header_sha256=sha(p/(a.variant+'.cuh')),
    stress_source_sha256=sha(p/'test_cofetch_progress.py'),
    helper_source_sha256=sha(root/'tests/attention/test_mla_host_batch.py'),
    start=time.strftime('%FT%T%z'), resources_start=require_exclusive(), complete=False,
    tests=6, performance_claim=False)
out = p/('stress_'+a.variant+'_gpu3.json')
out.write_text(json.dumps(record, indent=2)+'\n')
env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
    CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0',
    PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root), TMPDIR=str(root/'temp'),
    TUNE_VARIANT=a.variant, FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+a.variant+'_normal')),
    FLASHINFER_EXTRA_CUDAFLAGS='')
cmd = ['numactl', '-N1', '-m1', '/mnt/data/mhy-vllm-venv/bin/python', '-m', 'pytest',
       '-q', '-p', 'no:cacheprovider', '-p', 'jitpaths', '-p', 'candidatepaths',
       str(p/'test_cofetch_progress.py')]
record['command'] = cmd
log = p/('pytest_stress_'+a.variant+'.log')
with log.open('w') as stream:
    child = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record['pid'] = child.pid
    out.write_text(json.dumps(record, indent=2)+'\n')
    record['exit_code'] = child.wait()
record.update(end=time.strftime('%FT%T%z'), resources_end=require_exclusive(), log_sha256=sha(log))
record['complete'] = record['exit_code'] == 0
out.write_text(json.dumps(record, indent=2)+'\n')
print('STRESS_COMPLETE', a.variant, record['complete'], flush=True)
raise SystemExit(record['exit_code'])
