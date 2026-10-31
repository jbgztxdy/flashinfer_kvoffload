"""Force histogram work; original cold-cache sanitizer first launch can skip radix."""
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
out = p / 'radix_sanitizer_state.json'
state = dict(start=time.strftime('%FT%T%z'), complete=False, steps=[],
    reason='Original first cold-cache launch may have equal keys and skip radix. Instrument the next launch with injected four-byte stamp differences and 4096 idle slots.')
def save():
    out.write_text(json.dumps(state, indent=2) + '\n')
save()
while True:
    record = json.loads((p / 'validation_state.json').read_text())
    if record.get('complete'):
        break
    if record.get('phase') == 'failed':
        raise SystemExit('Original validation failed; inspect preserved state')
    time.sleep(10)
env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
    CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
    TMPDIR=str(root / 'temp'), MAX_JOBS='4', TUNE_VARIANT='warp_hist',
    FLASHINFER_WORKSPACE_BASE=str(p / 'jit_warp_hist_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
for kind in ['synccheck', 'racecheck']:
    cmd = ['numactl', '-N1', '-m1', '/usr/local/cuda-13.1/bin/compute-sanitizer', '--tool', kind,
        '--launch-timeout', '0', '--error-exitcode', '86', '--kernel-name',
        'regex=WarpHistogramHostParams', '--launch-skip', '1', '--launch-count', '1']
    if kind == 'racecheck':
        cmd += ['--racecheck-num-workers', '8']
    cmd += ['/mnt/data/mhy-vllm-venv/bin/python', '-u', str(p / 'probe_radix_warp_hist.py')]
    step = dict(tool=kind, command=cmd, start=time.strftime('%FT%T%z'))
    state['steps'].append(step)
    with (p / (kind + '_radix_warp_hist.log')).open('w') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        step['pid'] = child.pid
        save()
        step['exit_code'] = child.wait()
    step['end'] = time.strftime('%FT%T%z')
    save()
    if step['exit_code']:
        raise SystemExit(step['exit_code'])
state.update(complete=True, end=time.strftime('%FT%T%z'))
save()
