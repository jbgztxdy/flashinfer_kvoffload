"""Stop only this candidate's own matrix launchers after their current child exits.

Frozen inputs and completed/partial results stay in place. Never signal a GPU
benchmark child or any other user's task. SIGSTOP applies only to own CPU parent.
"""
import json
import os
import signal
import time
from pathlib import Path
p=Path(__file__).resolve().parent
expected=[['/mnt/data/mhy-vllm-venv/bin/python','-u','temp/glm_trace/tune_wait_20261031/validate_ids.py',
           '--hardware',hardware,'--cohort',cohort] for hardware,cohort in [('sxm','limited'),('sxm','native'),('gpu3','native')]]
targets=[]
for command in expected:
    matches=[]
    for f in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            actual=[x.decode() for x in f.read_bytes().split(b'\0') if x]
            if actual==command and f.stat().st_uid==os.getuid():
                matches.append(int(f.parent.name))
        except (OSError,UnicodeError):
            pass
    assert len(matches)<=1,(command,matches)
    if matches:
        pid=matches[0]
        os.kill(pid,signal.SIGSTOP)
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        targets.append(dict(pid=pid,command=command,children=list(map(int,children)),stopped=False))
record=dict(start=time.strftime('%FT%T%z'),reason='Original fetch_ids normal PCIe no-LRU medium regression reaches18.5%; preserve original plan/data, stop broad work before validating a cached-only specialization.',targets=targets)
out=p/'rejected_matrix_stop.json'
out.write_text(json.dumps(record,indent=2)+'\n')
while any(not x['stopped'] for x in targets):
    for t in targets:
        if t['stopped']:
            continue
        active=[]
        for pid in t['children']:
            f=Path(f'/proc/{pid}/stat')
            try:
                state=f.read_text().rsplit(') ',1)[1].split()[0]
            except FileNotFoundError:
                state='gone'
            if state not in ['Z','gone']:
                active.append(pid)
        if active:
            print('WAITING_OWN_BENCH_CHILD',t['pid'],active,flush=True)
            continue
        try:
            os.kill(t['pid'],signal.SIGTERM)
            os.kill(t['pid'],signal.SIGCONT)
        except ProcessLookupError:
            pass
        t['stopped']=True
        t['end']=time.strftime('%FT%T%z')
        print('STOPPED_OWN_PARENT_AFTER_CHILD',t['pid'],flush=True)
    out.write_text(json.dumps(record,indent=2)+'\n')
    if any(not x['stopped'] for x in targets):
        time.sleep(10)
record['end']=time.strftime('%FT%T%z')
out.write_text(json.dumps(record,indent=2)+'\n')
