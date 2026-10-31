"""Serialize the new diagnostic behind the existing GPU0/1 experiment lock."""
import json
import os
import subprocess
import time
from pathlib import Path
from resources import snapshot
p = Path(__file__).resolve().parent
root = p.parents[2]
while not list((p / 'jit_profile_wait').rglob('*host_overlap*.so')):
    print('WAITING_CPU_BUILD', flush=True)
    time.sleep(15)
while not snapshot()['available']:
    print('WAITING_GPU01', snapshot(), flush=True)
    time.sleep(30)
env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0,1',
    CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
    TMPDIR=str(root / 'temp'), MAX_JOBS='4', TUNE_VARIANT='profile_wait', TUNE_RUN_TAG='budget_',
    FLASHINFER_WORKSPACE_BASE=str(p / 'jit_profile_wait'), FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
cmd = ['numactl', '-N0', '-m0', '/mnt/data/mhy-vllm-venv/bin/python', '-u', str(p / 'bench.py'),
    '--kind', 'lru', '--mtp', '8', '--requests', '128', '--settings', 'g100_l20000', 'g50_l20000', 'nvlink',
    '--passes', '1', '--positions', '1024', '--fetch-ctas', '24', '--host-split', 'off']
state = dict(start=time.strftime('%FT%T%z'), command=cmd, classification='diagnostic, not performance gain')
with (p / 'diagnostic.log').open('a') as log:
    child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['pid'] = child.pid
    (p / 'diagnostic_state.json').write_text(json.dumps(state, indent=2) + '\n')
    print('DIAGNOSTIC_START', child.pid, flush=True)
    state['exit_code'] = child.wait()
state['end'] = time.strftime('%FT%T%z')
(p / 'diagnostic_state.json').write_text(json.dumps(state, indent=2) + '\n')
print('DIAGNOSTIC_DONE', state['exit_code'], flush=True)
raise SystemExit(state['exit_code'])
