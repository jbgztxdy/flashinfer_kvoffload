"""Check opposite full-trace order and native links before broadening shapes."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
py = '/mnt/data/mhy-vllm-venv/bin/python'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
read = lambda f: json.loads(f.read_text())
a = read(p / 'performance_audit.json')
full = [r for r in a['pairs'] if r['cohort'] == 'cachedreverse1024_f24_']
assert len(full) == 3 and all(r['reduction'] > .01 for r in full if r['setting'].startswith('g100'))
assert all(r['reduction'] > -.01 for r in full)
plan = dict(recorded_at=time.strftime('%FT%T%z'), positions=1024,passes=5,mtp=8,requests=128,
    fetch_ctas=24,host_split='off',order=['base','cached_pe'],
    headers={v:sha(p/(v+'.cuh')) for v in ['base','cached_pe']},
    tasks=[dict(tag='cachedforward1024_f24_',native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
           dict(tag='cachednative1024_f24_',native=True,settings=['native_hbm','pcie'])],
    reason='Reversefull100GB+1.471/+1.323%; oppositeorderandnativechecksneeded. GPU3nativeHBMisnotNVLink. No adoption before cross-shape/SXM gates.')
(p/'extension_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
state=dict(start=time.strftime('%FT%T%z'),complete=False,candidate_adopted=False,goal_complete=False,steps=[])
out=p/'extension_state.json'
def save():out.write_text(json.dumps(state,indent=2)+'\n')
save()
for task in plan['tasks']:
    cmd=[py,'-u',str(p/'screen.py'),'--hardware','gpu3','--candidate','cached_pe',
        '--variants',*plan['order'],'--positions','1024','--passes','5','--requests','128',
        '--mtp','8','--fetch-ctas','24','--host-split','off','--tag',task['tag'],'--settings',*task['settings']]
    if task['native']:cmd+=['--native']
    step=dict(command=cmd,start=time.strftime('%FT%T%z'),task=task)
    state['steps'].append(step)
    with (p/(task['tag']+'screen.log')).open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        step['pid']=child.pid;save()
        step['exit_code']=child.wait()
    step['end']=time.strftime('%FT%T%z');save()
    if step['exit_code']:raise SystemExit(step['exit_code'])
    subprocess.run([py,str(p/'audit_performance.py')],cwd=root,stdout=(p/'audit_performance.log').open('w'),check=True)
state.update(complete=True,end=time.strftime('%FT%T%z'));save()
print('EXTENSION_COMPLETE',flush=True)
