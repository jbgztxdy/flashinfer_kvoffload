"""Continue original builds/codegen, then correctness, real dispatch and sanitizer gates."""
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
py = '/mnt/data/mhy-vllm-venv/bin/python'
read = lambda f: json.loads(f.read_text())
state = dict(start=time.strftime('%FT%T%z'), phase='waiting_original_normal_and_emu_codegen',
    complete=False, candidate_adopted=False, goal_complete=False, steps=[])
out = p / 'validation_state.json'
def save(): out.write_text(json.dumps(state, indent=2) + '\n')
def run(cmd, name, env=None):
    record = dict(command=cmd, start=time.strftime('%FT%T%z'), log=name)
    state['steps'].append(record)
    with (p / name).open('a') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid; save(); record['exit_code'] = child.wait()
    record['end'] = time.strftime('%FT%T%z'); save()
    if record['exit_code']: raise RuntimeError('Failed ' + name)
save()
try:
    while True:
        t = read(p / 'tests_nine_warp_gpu3.json')
        if t.get('compile_exit_code') not in (None, 0) or t.get('exit_code') not in (None, 0):
            raise RuntimeError('Original normal compile/test failed')
        proof = p / 'compiler_nine_warp_summary.json'
        if t.get('complete') and proof.exists():
            assert read(proof)['complete'] and read(proof)['unchanged_original_kernels'] == 36
            break
        if 'Traceback' in (p / 'compiler_nine_warp.log').read_text():
            raise RuntimeError('Original EMU codegen failed; inspect original23187 before any retry')
        time.sleep(10)
    state['phase'] = 'normal_codegen_long_payload_dispatch'; save()
    run([py, str(p / 'compiler_nine_warp.py'), '--normal'], 'compiler_nine_warp_normal.log')
    assert read(p / 'compiler_nine_warp_normal_summary.json')['unchanged_original_kernels'] == 72
    run([py, '-u', str(p / 'stress_candidate.py'), '--variant', 'nine_warp'], 'stress_nine_warp.log')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
        CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
        TMPDIR=str(root / 'temp'), MAX_JOBS='4',
        FLASHINFER_WORKSPACE_BASE=str(p / 'jit_nine_warp_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
    prefix = ['numactl', '-N1', '-m1']
    run(prefix + [py, '-u', str(p / 'probe_nine_warp.py')], 'probe_nine_warp.log', env)
    state['phase'] = 'sanitizers'; save()
    for kind in ['synccheck', 'racecheck']:
        cmd = prefix + ['/usr/local/cuda-13.1/bin/compute-sanitizer', '--tool', kind,
            '--launch-timeout', '0', '--error-exitcode', '86']
        if kind == 'racecheck': cmd += ['--kernel-name', 'regex=BatchMLANineWarpKernel', '--launch-count', '1', '--racecheck-num-workers', '8']
        cmd += [py, '-u', str(p / ('probe_' + kind + '_nine_warp.py'))]
        run(cmd, kind + '_nine_warp.log', env)
        assert 'SUMMARY: 0' in (p / (kind + '_nine_warp.log')).read_text()
    state['phase'] = 'short_screen'; save()
    run([py, '-u', str(p / 'screen.py'), '--hardware', 'gpu3', '--candidate', 'nine_warp',
        '--variants', 'base', 'nine_warp', '--tag', 'nine256_f24_'], 'screen_gpu3.log')
    state.update(phase='short_screen_complete', complete=True, end=time.strftime('%FT%T%z')); save()
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z')); save(); raise
