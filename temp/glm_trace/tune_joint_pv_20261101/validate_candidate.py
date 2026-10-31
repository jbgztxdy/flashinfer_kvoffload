"""Finish the original build/test handles, then gate one fixed three-way screen."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
py = '/mnt/data/mhy-vllm-venv/bin/python'
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
state = dict(start=time.strftime('%FT%T%z'), phase='waiting_existing_compile_and_tests',
             complete=False, candidate_adopted=False, goal_complete=False, steps=[])
out = p / 'validation_state.json'
def save():
    out.write_text(json.dumps(state, indent=2) + '\n')
def run(cmd, name, env=None):
    record = dict(command=cmd, start=time.strftime('%FT%T%z'), log=name)
    state['steps'].append(record)
    with (p / name).open('a') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        save()
        record['exit_code'] = child.wait()
    record['end'] = time.strftime('%FT%T%z')
    save()
    if record['exit_code']:
        raise RuntimeError('Failed ' + name)
save()
try:
    while True:
        t = read(p / 'tests_joint_pv_gpu3.json')
        if t.get('compile_exit_code') not in (None, 0) or t.get('exit_code') not in (None, 0):
            raise RuntimeError('Original normal compile/correctness handle failed')
        emu = p / 'compile_joint_pv_emu.log'
        if 'COMPILED joint_pv' in emu.read_text() and t.get('complete'):
            break
        processes = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
        live = [s for s in processes if 'python -u' in s and str(p.relative_to(root) / 'compile.py') in s]
        if 'COMPILED joint_pv' not in emu.read_text() and not live:
            raise RuntimeError('Original EMU build no longer live and no successful marker')
        time.sleep(10)
    state['phase'] = 'codegen_and_long_payload_and_graph'
    save()
    for normal in [False, True]:
        run([py, str(p / 'compiler_joint_pv.py')] + (['--normal'] if normal else []),
            'compiler_joint_pv' + ('_normal' if normal else '') + '.log')
    run([py, '-u', str(p / 'stress_candidate.py'), '--variant', 'joint_pv'], 'stress_joint_pv.log')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
        CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
        TMPDIR=str(root / 'temp'), MAX_JOBS='4',
        FLASHINFER_WORKSPACE_BASE=str(p / 'jit_joint_pv_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
    run(['numactl', '-N1', '-m1', py, '-u', str(p / 'probe_joint_pv.py')], 'probe_joint_pv.log', env)
    state['phase'] = 'fixed_threeway_short_screen'
    save()
    plan = dict(positions=256, passes=3, mtp=8, requests=128, fetch_ctas=24,
        settings=['g100_l0', 'g100_l20000', 'g50_l20000'],
        order=['joint_qk', 'joint_pv', 'base'],
        headers={v: sha(p / (v + '.cuh')) for v in ['base', 'joint_qk', 'joint_pv']},
        mechanism=read(p / 'manifest.json')['mechanism'])
    f = p / 'short_plan.json'
    if f.exists():
        assert read(f) == plan
    else:
        f.write_text(json.dumps(plan, indent=2) + '\n')
    run([py, '-u', str(p / 'screen.py'), '--hardware', 'gpu3', '--candidate', 'joint_pv',
         '--variants', *plan['order'], '--tag', 'publish256_f24_'], 'screen_gpu3.log')
    state.update(phase='short_screen_complete', complete=True, end=time.strftime('%FT%T%z'))
    save()
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
