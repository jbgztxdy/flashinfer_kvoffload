"""Run an isolated FP32 expression diagnostic, never an attention timing/reference."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
from resources_gpu3 import snapshot, require_exclusive
assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
assert 'membind: 1' in subprocess.check_output(['numactl','--show'],text=True)
lock = (p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3',snapshot(),flush=True)
    time.sleep(10)
before = require_exclusive()
binary = p/'probe_scale_roundoff.bin'
values = json.loads(subprocess.check_output([str(binary)],text=True))
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
record = dict(timestamp=time.strftime('%FT%T%z'),values=values,resources_start=before,
    resources_end=require_exclusive(),source_sha256=sha(p/'probe_scale_roundoff.cu'),
    binary_sha256=sha(binary),complete=True,performance_claim=False,reference_claim=False,
    compiler_flags='nvcc13.1 -O3 --use_fast_math -arch=sm_80 -std=c++17',
    limitation='Isolated equal-max expression shows possible contraction roundoff; it does not measure real attention max-stability or branch frequencies. Primary kernel SASS must separately establish its arithmetic pattern.')
(p/'scale_roundoff.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
