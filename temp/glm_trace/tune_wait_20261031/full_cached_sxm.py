"""Retry the two full confirmations whose previous attempts failed before timing."""
import json
import subprocess
import sys
import time
from pathlib import Path
p=Path(__file__).resolve().parent
state=[]
for native in [False,True]:
    command=[sys.executable,'-u',str(p/'screen.py'),'--hardware','sxm','--candidate','cached_ids',
             '--variants',*(['base','cached_ids'] if native else ['cached_ids','base']),
             '--positions','1024','--passes','5','--fetch-ctas','24',
             '--tag','cached_native_full_' if native else 'cached_full_']
    if native:command+=['--native','--settings','nvlink','pcie']
    item=dict(command=command,start=time.strftime('%FT%T%z'));state.append(item)
    with (p/('full_cached_retry_'+('native' if native else 'emu')+'.log')).open('a') as log:
        child=subprocess.Popen(command,cwd=p.parents[2],stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid
        (p/'full_cached_retry_state.json').write_text(json.dumps(state,indent=2)+'\n')
        item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z')
    (p/'full_cached_retry_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if item['exit_code']:raise SystemExit(item['exit_code'])
