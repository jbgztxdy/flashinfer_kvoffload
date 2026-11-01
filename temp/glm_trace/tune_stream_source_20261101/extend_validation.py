"""Run complete regression only after the sole full local producer passes; never install."""
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
out = p / 'extension_state.json'
assert not out.exists(), 'Inspect existing producer; do not duplicate'
state = dict(pid=os.getpid(), start=time.strftime('%FT%T%z'), phase='waiting_original_local_confirmation',
             complete=False, goal_complete=False, candidate_adopted=False, steps=[])
def save():
    temporary = out.with_suffix('.json.new')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(out)
def run(cmd, logname):
    item = dict(command=cmd, log=logname, start=time.strftime('%FT%T%z'))
    state['steps'].append(item)
    with (p / logname).open('a') as log:
        child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        item['pid'] = child.pid
        save()
        item['exit_code'] = child.wait()
    item['end'] = time.strftime('%FT%T%z')
    save()
    if item['exit_code']:
        raise RuntimeError('Failed ' + logname)
def refresh():
    for name in ['audit_performance.py', 'audit_coverage.py', 'report.py']:
        run([py, str(p / name)], 'extension_' + name.removesuffix('.py') + '.log')
def gate(tag, settings):
    rows = [r for r in read(p / 'performance_audit.json')['pairs'] if r['cohort'] == tag]
    assert len(rows) == len(settings) and {r['setting'] for r in rows} == set(settings)
    assert all(r['positions'] == 1024 and r['passes'] == 5 for r in rows)
    assert all(r['reduction'] >= -.01 for r in rows), (tag, rows)
    assert all(r['reduction'] > 0 for r in rows if r['setting'].startswith('g100')), (tag, rows)
save()
try:
    while True:
        original = read(p / 'confirmation_state.json')
        if original['complete']:
            assert original['phase'] == 'local_confirmation_complete', 'Local confirmation did not pass; preserve rejection'
            break
        assert original['phase'] != 'failed', 'Original confirmation failed; inspect preserved evidence'
        ps = subprocess.run(['ps', '-p', str(original['pid']), '-o', 'pid,args'], text=True, capture_output=True)
        assert ps.returncode == 0 and 'tune_stream_source_20261101/confirm_if_promising.py' in ps.stdout
        state.update(last_observation=time.strftime('%FT%T%z'), original_pid=original['pid'], original_phase=original['phase'])
        save()
        time.sleep(20)
    assert sha(p / 'stream_source.cuh') == '48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73'
    for tag in ['streamsource_reverse1024_f24_', 'streamsource_forward1024_f24_']:
        gate(tag, ['g100_l0', 'g100_l20000', 'g50_l20000'])
    gate('streamsource_native1024_f24_', ['native_hbm', 'pcie'])
    for script in ['validate_shapes.py', 'validate_sxm.py']:
        for cohort in ['limited', 'native']:
            run([py, str(p / script), '--cohort', cohort, '--plan-only'], script.removesuffix('.py') + '_' + cohort + '_plan.log')
    state['phase'] = 'local_matrix'
    save()
    for cohort in ['limited', 'native']:
        run([py, '-u', str(p / 'validate_shapes.py'), '--cohort', cohort], 'matrix_gpu3_' + cohort + '.log')
    refresh()
    state['gpu3_matrix_complete'] = True
    state['phase'] = 'sxm_full_confirmation'
    save()
    tasks = [dict(tag='streamsource_sxm_reverse1024_f24_', variants=['stream_source', 'base'], settings=['g100_l0', 'g100_l20000', 'g50_l20000'], native=False),
             dict(tag='streamsource_sxm_forward1024_f24_', variants=['base', 'stream_source'], settings=['g100_l0', 'g100_l20000', 'g50_l20000'], native=False),
             dict(tag='streamsource_sxm_native1024_f24_', variants=['base', 'stream_source'], settings=['nvlink', 'pcie'], native=True)]
    (p / 'sxm_confirmation_plan.json').write_text(json.dumps(dict(recorded_at=time.strftime('%FT%T%z'),
        positions=1024, passes=5, requests=128, mtp=8, fetch_ctas=24, host_split='off',
        base_sha256=sha(p / 'base.cuh'), candidate_sha256=sha(p / 'stream_source.cuh'), tasks=tasks), indent=2) + '\n')
    for task in tasks:
        cmd = [py, '-u', str(p / 'screen.py'), '--hardware', 'sxm', '--candidate', 'stream_source',
               '--variants', *task['variants'], '--positions', '1024', '--passes', '5', '--requests', '128',
               '--mtp', '8', '--fetch-ctas', '24', '--host-split', 'off', '--settings', *task['settings'], '--tag', task['tag']]
        if task['native']:
            cmd += ['--native']
        run(cmd, task['tag'] + 'screen.log')
        run([py, str(p / 'audit_performance.py')], 'sxm_confirmation_audit.log')
        gate(task['tag'], task['settings'])
    state.update(phase='sxm_matrix', sxm_confirmation_gate_passed=True)
    save()
    for cohort in ['limited', 'native']:
        run([py, '-u', str(p / 'validate_sxm.py'), '--cohort', cohort], 'matrix_sxm_' + cohort + '.log')
    refresh()
    coverage = read(p / 'coverage.json')
    assert coverage['gpu3_matrix_complete'] and coverage['sxm_matrix_complete']
    state.update(phase='regression_measurements_complete', complete=True, end=time.strftime('%FT%T%z'),
                 remaining=['Every negative regression needs controls/review; frozen retention and fresh root validation before installation.'])
    save()
    print('REGRESSION_MEASUREMENTS_COMPLETE_NO_INSTALL', flush=True)
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
