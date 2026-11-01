"""Expand only a successful paired short screen; preserve the sole original producer."""
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
out = p / 'confirmation_state.json'
assert not out.exists(), 'Inspect the existing producer instead of duplicating it'
state = dict(pid=os.getpid(), start=time.strftime('%FT%T%z'), phase='waiting_original_short_screen',
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
save()
try:
    while True:
        original = read(p / 'validation_state.json')
        if original['phase'] == 'failed':
            raise RuntimeError('Original short producer failed; inspect its preserved evidence')
        if original['complete']:
            break
        ps = subprocess.run(['ps', '-p', str(original['pid']), '-o', 'pid,args'], text=True, capture_output=True)
        assert ps.returncode == 0 and 'tune_local_order_20261101/validate_candidate.py' in ps.stdout
        state.update(last_observation=time.strftime('%FT%T%z'), original_phase=original['phase'], original_pid=original['pid'])
        save()
        time.sleep(20)
    a = read(p / 'performance_audit.json')
    assert len(a['pairs']) == 3 and len(a['rows']) == 6
    plan = dict(recorded_at=time.strftime('%FT%T%z'), base_sha256=sha(p / 'base.cuh'),
                candidate_sha256=sha(p / 'local_order.cuh'), short_audit_sha256=sha(p / 'performance_audit.json'),
                gate='Both100GB short reductions>0; every setting>=-1%. Full confirmation same rule. No installation.',
                positions=1024, passes=5, mtp=8, requests=128, fetch_ctas=24, host_split='off',
                tasks=[dict(tag='localorder_reverse1024_f24_', variants=['local_order', 'base'], settings=['g100_l0', 'g100_l20000', 'g50_l20000'], native=False),
                       dict(tag='localorder_forward1024_f24_', variants=['base', 'local_order'], settings=['g100_l0', 'g100_l20000', 'g50_l20000'], native=False),
                       dict(tag='localorder_native1024_f24_', variants=['base', 'local_order'], settings=['native_hbm', 'pcie'], native=True)])
    (p / 'confirmation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (p / 'short_performance_audit.json').write_bytes((p / 'performance_audit.json').read_bytes())
    def passes(rows):
        return all(x['reduction'] >= -.01 for x in rows) and all(x['reduction'] > 0 for x in rows if x['setting'].startswith('g100'))
    if not passes(a['pairs']):
        state.update(phase='rejected_short_screen', complete=True, end=time.strftime('%FT%T%z'))
        save()
        print('REJECT_SHORT_SCREEN_NO_EXPANSION', flush=True)
    else:
        for task in plan['tasks']:
            state['phase'] = task['tag']
            save()
            cmd = [py, '-u', str(p / 'screen.py'), '--hardware', 'gpu3', '--candidate', 'local_order',
                   '--variants', *task['variants'], '--positions', '1024', '--passes', '5', '--requests', '128',
                   '--mtp', '8', '--fetch-ctas', '24', '--host-split', 'off', '--settings', *task['settings'], '--tag', task['tag']]
            if task['native']:
                cmd += ['--native']
            run(cmd, task['tag'] + 'screen.log')
            run([py, str(p / 'audit_performance.py')], 'audit_confirmation.log')
            rows = [r for r in read(p / 'performance_audit.json')['pairs'] if r['cohort'] == task['tag']]
            assert len(rows) == len(task['settings'])
            if not passes(rows):
                state.update(phase='rejected_confirmation', rejected_cohort=task['tag'], complete=True, end=time.strftime('%FT%T%z'))
                save()
                print('REJECT_CONFIRMATION_NO_EXPANSION', task['tag'], flush=True)
                break
        else:
            state.update(phase='local_confirmation_complete', complete=True, end=time.strftime('%FT%T%z'),
                         remaining=['Full156 matrix, actualNVLink/SXM confirmations and negativecontrols required before any retention.'])
            save()
            print('LOCAL_CONFIRMATION_COMPLETE_NO_INSTALL', flush=True)
except Exception as exc:
    state.update(phase='failed', error=str(exc), end=time.strftime('%FT%T%z'))
    save()
    raise
