"""Resume only missing native rows after the preserved resource preflight rejection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
failure = read(p / 'resource_preflight_failure.json')
original = read(p / 'broader_validation_state.json')
assert original['phase'] == 'failed' and original['error'] == 'Failed matrix_sxm_native.log'
assert original['steps'][-2]['exit_code'] == 0 and original['steps'][-1]['exit_code'] == 1
for pid in failure['original_pids'].values():
    assert subprocess.run(['ps', '-p', str(pid)], stdout=subprocess.DEVNULL).returncode != 0
for item in failure['preserved']:
    assert sha(p / 'resource_preflight_failure' / item['name']) == item['sha256']
out = p / 'matrix_resume_state.json'
assert not out.exists(), 'Inspect existing resume producer instead of duplicating it'
state = dict(pid=os.getpid(), start=time.strftime('%FT%T%z'), phase='native_matrix', complete=False,
             goal_complete=False, candidate_adopted=False, steps=[],
             preserved_failure_sha256=sha(p / 'resource_preflight_failure.json'))
def save():
    temp = out.with_suffix('.json.new')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(out)
save()
py = '/mnt/data/mhy-vllm-venv/bin/python'
try:
    tasks = [([py, '-u', str(p / 'validate_sxm.py'), '--cohort', 'native'], 'matrix_sxm_native_resume.log')]
    tasks += [([py, str(p / name)], 'resume_' + name.removesuffix('.py') + '.log')
              for name in ['audit_performance.py', 'audit_coverage.py', 'update_decision.py', 'report.py']]
    for cmd, name in tasks:
        item = dict(command=cmd, log=name, start=time.strftime('%FT%T%z'))
        state['steps'].append(item)
        with (p / name).open('a') as log:
            child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            item['pid'] = child.pid
            save()
            item['exit_code'] = child.wait()
        item['end'] = time.strftime('%FT%T%z')
        save()
        if item['exit_code']:
            raise RuntimeError('Failed ' + name)
    coverage = read(p / 'coverage.json')
    assert coverage['gpu3_matrix_complete'] and coverage['sxm_matrix_complete']
    state.update(phase='regression_measurements_complete', complete=True, end=time.strftime('%FT%T%z'))
    save()
    print('RESUMED_NATIVE_MATRIX_COMPLETE_NO_INSTALL', flush=True)
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
