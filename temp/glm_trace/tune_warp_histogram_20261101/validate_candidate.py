"""Observe original build/test handles, then run one fixed two-shape screen."""
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
state = dict(start=time.strftime('%FT%T%z'), phase='waiting_existing_build_test_stress',
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
        t = read(p / 'tests_warp_hist_gpu3.json')
        st = read(p / 'stress_warp_hist_gpu3.json') if (p / 'stress_warp_hist_gpu3.json').exists() else {}
        if any(x.get(k) not in (None, 0) for x in [t, st] for k in ['compile_exit_code', 'exit_code']):
            raise RuntimeError('Original build/test/stress failed; inspect original records')
        if ('COMPILED warp_hist' in (p / 'compile_warp_hist_emu.log').read_text()
                and t.get('complete') and st.get('complete')):
            break
        time.sleep(10)
    state['phase'] = 'codegen_words_graph_sanitizers'
    save()
    for normal in [False, True]:
        run([py, str(p / 'compiler_warp_hist.py')] + (['--normal'] if normal else []),
            'compiler_warp_hist' + ('_normal' if normal else '') + '.log')
    run([py, str(p / 'verify_machine_words.py')], 'verify_machine_words.log')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
        CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
        TMPDIR=str(root / 'temp'), MAX_JOBS='4', TUNE_VARIANT='warp_hist',
        FLASHINFER_WORKSPACE_BASE=str(p / 'jit_warp_hist_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
    prefix = ['numactl', '-N1', '-m1']
    run(prefix + [py, '-u', str(p / 'probe_warp_hist.py')], 'probe_warp_hist.log', env)
    for kind in ['synccheck', 'racecheck']:
        cmd = prefix + ['/usr/local/cuda-13.1/bin/compute-sanitizer', '--tool', kind,
            '--launch-timeout', '0', '--error-exitcode', '86', '--kernel-name',
            'regex=WarpHistogramHostParams', '--launch-count', '1']
        if kind == 'racecheck':
            cmd += ['--racecheck-num-workers', '8']
        run(cmd + [py, '-u', str(p / 'probe_warp_hist.py')], kind + '_warp_hist.log', env)
    plan = dict(positions=256, passes=3, shapes=[dict(mtp=1, requests=1), dict(mtp=8, requests=1)],
        fetch_ctas=16, host_split='on', settings=['g100_l0','g100_l20000','g50_l20000'],
        order=['base','warp_hist'], headers={v:sha(p/(v+'.cuh')) for v in ['base','warp_hist']})
    (p / 'short_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    state['phase'] = 'fixed_two_shape_short_screen'
    save()
    for shape in plan['shapes']:
        run([py, '-u', str(p / 'screen.py'), '--hardware', 'gpu3', '--candidate', 'warp_hist',
            '--variants', *plan['order'], '--requests', str(shape['requests']), '--mtp', str(shape['mtp']),
            '--fetch-ctas','16','--host-split','on','--tag',f"small256_m{shape['mtp']}_"],
            f"screen_mtp{shape['mtp']}_gpu3.log")
    run([py, str(p / 'audit_performance.py')], 'audit_performance.log')
    state.update(phase='short_screen_complete', complete=True, end=time.strftime('%FT%T%z'))
    save()
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
