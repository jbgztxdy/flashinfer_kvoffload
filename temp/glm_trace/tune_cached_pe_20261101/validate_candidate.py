"""Gate one fixed same-parameter large-batch QPE reuse experiment."""
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
state = dict(start=time.strftime('%FT%T%z'), phase='waiting_original_build_test_stress',
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
        t = read(p / 'tests_cached_pe_gpu3.json')
        st = read(p / 'stress_cached_pe_gpu3.json') if (p / 'stress_cached_pe_gpu3.json').exists() else {}
        if any(x.get(k) not in (None, 0) for x in [t, st] for k in ['compile_exit_code', 'exit_code']):
            raise RuntimeError('Original build/test/stress failed; preserve original records')
        if ('COMPILED cached_pe' in (p / 'compile_cached_pe_emu.log').read_text()
                and t.get('complete') and st.get('complete')):
            break
        time.sleep(10)
    state['phase'] = 'codegen_words_graph_sanitizers'
    save()
    for normal in [False, True]:
        run([py, str(p / 'compiler_cached_pe.py')] + (['--normal'] if normal else []),
            'compiler_cached_pe' + ('_normal' if normal else '') + '.log')
    run([py, str(p / 'verify_machine_words.py')], 'verify_machine_words.log')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3',
        CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
        TMPDIR=str(root / 'temp'), MAX_JOBS='4', TUNE_VARIANT='cached_pe',
        FLASHINFER_WORKSPACE_BASE=str(p / 'jit_cached_pe_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
    prefix = ['numactl', '-N1', '-m1']
    run(prefix + [py, '-u', str(p / 'probe_cached_pe.py')], 'probe_cached_pe.log', env)
    for kind in ['synccheck', 'racecheck']:
        cmd = prefix + ['/usr/local/cuda-13.1/bin/compute-sanitizer', '--tool', kind,
            '--launch-timeout', '0', '--error-exitcode', '86', '--kernel-name',
            'regex=BatchMLACachedPEKernel', '--launch-count', '1']
        if kind == 'racecheck': cmd += ['--racecheck-num-workers', '8']
        run(cmd + [py, '-u', str(p / 'probe_cached_pe.py')], kind + '_cached_pe.log', env)
    plan = dict(positions=256, passes=3, mtp=8, requests=128, fetch_ctas=24, host_split='off',
        settings=['g100_l0','g100_l20000','g50_l20000'], order=['base','cached_pe'],
        headers={v:sha(p/(v+'.cuh')) for v in ['base','cached_pe']},
        unroll='Frozen original4, no numerical sweep',mechanism=read(p/'manifest.json')['mechanism'])
    (p / 'short_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    state['phase'] = 'fixed_large_short_screen'
    save()
    run([py, '-u', str(p / 'screen.py'), '--hardware','gpu3','--candidate','cached_pe',
        '--variants',*plan['order'],'--requests','128','--mtp','8','--fetch-ctas','24',
        '--host-split','off','--tag','cached256_f24_'], 'screen_gpu3.log')
    run([py, str(p / 'audit_performance.py')], 'audit_performance.log')
    state.update(phase='short_screen_complete', complete=True, end=time.strftime('%FT%T%z'))
    save()
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
