"""Resume incomplete cached-only cohorts serially; preserve all interrupted records."""
import json
import subprocess
import sys
import time
from pathlib import Path

p=Path(__file__).resolve().parent
state=[]
commands=[['validate_cached_ids.py','--hardware','sxm','--cohort',cohort] for cohort in ['limited','native']]
for args in commands:
    item=dict(command=[sys.executable,'-u',str(p/args[0]),*args[1:]], start=time.strftime('%FT%T%z'))
    state.append(item)
    with (p/('resume_cached_sxm_'+args[-1]+'.log')).open('a') as log:
        child=subprocess.Popen(item['command'],cwd=p.parents[2],stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid
        (p/'resume_cached_sxm_state.json').write_text(json.dumps(state,indent=2)+'\n')
        item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z')
    (p/'resume_cached_sxm_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if item['exit_code']:raise SystemExit(item['exit_code'])
