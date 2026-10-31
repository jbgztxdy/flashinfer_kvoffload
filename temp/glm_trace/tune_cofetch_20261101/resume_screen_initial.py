"""Resume only missing candidate cells; retry confirmed foreign-GPU preflight failures."""
import json
import os
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
from resources_gpu3 import snapshot
state = dict(pid=os.getpid(), start=time.strftime('%FT%T%z'), attempts=[], complete=False)
out = p/'resume_screen_state.json'
def save():
    out.write_text(json.dumps(state, indent=2)+'\n')
save()
for variant in ['cofetch', 'stage1_control']:
    while True:
        # Five quiet samples reduce the import/preflight race; the immutable
        # benchmark still performs its own exclusive checks and owns the lock.
        quiet = 0
        while quiet < 5:
            observation = snapshot()
            state['last_observation'] = observation
            quiet = quiet + 1 if observation['available'] else 0
            state['quiet_samples'] = quiet
            save()
            time.sleep(2 if quiet else 15)
        log = p/('cofetch256_f24_gpu3_'+variant+'.log')
        offset = log.stat().st_size if log.exists() else 0
        item = dict(variant=variant, start=time.strftime('%FT%T%z'))
        state['attempts'].append(item)
        cmd = ['/mnt/data/mhy-vllm-venv/bin/python', '-u', str(p/'screen.py'),
            '--hardware', 'gpu3', '--candidate', 'cofetch', '--variants', variant,
            '--tag', 'cofetch256_f24_', '--fetch-ctas', '24']
        with (p/('resume_'+variant+'.log')).open('a') as stream:
            child = subprocess.Popen(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            item['pid'] = child.pid
            save()
            item['exit_code'] = child.wait()
        item['end'] = time.strftime('%FT%T%z')
        save()
        if item['exit_code'] == 0:
            break
        text = log.read_text()[offset:] if log.exists() else ''
        preflight = ('Other physical GPU3 process:' in text and
                     ('RESOURCE_START=require_exclusive()' in text or
                      'run_setting' in text and 'require_exclusive()' in text))
        item['confirmed_foreign_preflight_failure'] = preflight
        save()
        if not preflight:
            raise SystemExit('Non-preflight failure preserved; inspect '+str(log))
        print('CONFIRMED_PREFLIGHT_RETRY', variant, flush=True)
        time.sleep(15)
state.update(complete=True, end=time.strftime('%FT%T%z'))
save()
print('MISSING_SCREENS_COMPLETE', flush=True)
