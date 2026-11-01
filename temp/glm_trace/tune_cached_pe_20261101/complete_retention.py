"""Finish authorized retention after the original producer and strict evidence gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--after-matrix-resume', action='store_true')
args = parser.parse_args()
py = '/mnt/data/mhy-vllm-venv/bin/python'
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
production = root / 'include/flashinfer/attention/mla_host_batch.cuh'
candidate = sha(p / 'cached_pe.cuh')
base = sha(p / 'production_before_cached_pe.cuh')
assert base == '7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
state = dict(start=time.strftime('%FT%T%z'), pid=os.getpid(), complete=False, candidate_adopted=False,
             goal_complete=False, phase='waiting_original_15217', steps=[])
out = p / ('completion_resume_state.json' if args.after_matrix_resume else 'completion_state.json')
assert not out.exists(), 'Inspect existing completion state instead of starting another producer'
def save():
    temporary = out.with_suffix('.json.new')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(out)
def run(cmd, logname, env=None):
    item = dict(command=cmd, log=logname, start=time.strftime('%FT%T%z'))
    state['steps'].append(item)
    with (p / logname).open('a') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        item['pid'] = child.pid
        save()
        item['exit_code'] = child.wait()
    item['end'] = time.strftime('%FT%T%z')
    save()
    if item['exit_code']:
        raise RuntimeError('Failed ' + logname + ': ' + str(item['exit_code']))
save()
try:
    while True:
        original = read(p / ('matrix_resume_state.json' if args.after_matrix_resume else 'broader_validation_state.json'))
        if original['complete']:
            assert original['phase'] == 'regression_measurements_complete'
            break
        assert original['phase'] != 'failed', 'Original producer failed; inspect preserved evidence'
        ps = subprocess.run(['ps', '-p', str(original['pid']), '-o', 'pid,args'], text=True, capture_output=True)
        expected_producer = 'resume_matrix.py' if args.after_matrix_resume else 'validate_more.py'
        assert ps.returncode == 0 and ('tune_cached_pe_20261101/' + expected_producer) in ps.stdout
        state.update(last_observation=time.strftime('%FT%T%z'), original_phase=original['phase'], original_pid=original['pid'])
        save()
        print('WAITING_ORIGINAL', original['pid'], original['phase'], flush=True)
        time.sleep(30)
    state['phase'] = 'final_measurement_audit'
    save()
    for script in ['audit_performance.py', 'audit_coverage.py', 'verify_machine_words.py']:
        run([py, str(p / script)], 'completion_' + script.removesuffix('.py') + '.log')
    run([py, str(p / 'retain_cached_pe.py')], 'retention_dry_run.log')
    state['phase'] = 'install_exact_measured_header'
    save()
    run([py, str(p / 'retain_cached_pe.py'), '--install'], 'retention_install.log')
    assert sha(production) == candidate
    state.update(candidate_adopted=True, phase='fresh_root_validation')
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0',
               PYTHONPATH=str(root), TMPDIR=str(root / 'temp'), MAX_JOBS='4',
               FLASHINFER_WORKSPACE_BASE=str(p / 'jit_production_cached_pe_normal'), FLASHINFER_EXTRA_CUDAFLAGS='')
    env.pop('FLASHINFER_FAR_EMU', None)
    run([py, '-u', str(p / 'compile_production.py')], 'compile_production.log', env)
    run([py, '-u', str(p / 'test_production.py')], 'test_production.log')
    run([py, str(p / 'verify_production_codegen.py')], 'verify_production_codegen.log')
    probe_env = dict(env, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3')
    run(['numactl', '-N1', '-m1', py, '-u', str(p / 'probe_production_cached_pe.py')], 'probe_production_cached_pe.log', probe_env)
    test = read(p / 'tests_production_gpu3.json')
    assert test['complete'] and test['expected_tests'] == 111 and test['exit_code'] == 0
    assert '111 passed' in (p / 'pytest_production_gpu3.log').read_text()
    record = read(p / 'cached_pe_retention.json')
    record['production_validation'] = {name: sha(p / name) for name in
        ['tests_production_gpu3.json', 'production_codegen_normal_summary.json', 'dispatch_probe_production_cached_pe.json']}
    (p / 'cached_pe_retention.json').write_text(json.dumps(record, indent=2) + '\n')
    state.update(phase='complete', complete=True, candidate_adopted=True, end=time.strftime('%FT%T%z'))
    save()
    print('RETENTION_AND_FRESH_ROOT_VALIDATION_COMPLETE', candidate, flush=True)
except Exception as exc:
    state.update(phase='failed', complete=False, error=str(exc), end=time.strftime('%FT%T%z'))
    if sha(production) == candidate:
        production.write_bytes((p / 'production_before_cached_pe.cuh').read_bytes())
        state.update(rolled_back_to_base=True, candidate_adopted=False)
        f = p / 'cached_pe_retention.json'
        if f.exists():
            record = read(f)
            record.update(candidate_adopted=False, post_install_failed=str(exc), production_rolled_back_sha256=base)
            f.write_text(json.dumps(record, indent=2) + '\n')
    save()
    raise
