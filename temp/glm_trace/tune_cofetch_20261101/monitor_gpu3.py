"""Read-only one-second app samples while the existing short-screen process lives."""
import argparse
import json
from pathlib import Path
import time

p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--screen-pid', type=int, required=True)
a = parser.parse_args()
from resources_gpu3 import snapshot
process = Path('/proc')/str(a.screen_pid)
original = (process/'cmdline').read_bytes()
assert any(name in original for name in [b'tune_cofetch_20261101/screen.py',
                                        b'tune_cofetch_20261101/resume_screen.py'])
samples = conflicts = 0
start = time.strftime('%FT%T%z')
with (p/'gpu3_monitor.jsonl').open('x') as stream:
    while True:
        try:
            if (process/'cmdline').read_bytes() != original:
                break
        except FileNotFoundError:
            break
        state = snapshot(allowed_pid=-1)
        ours, foreign = [], []
        for app in state.pop('foreign'):
            try:
                command = (Path('/proc')/str(app['pid'])/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError:
                command = ''
            app['command'] = command
            (ours if 'tune_cofetch_20261101/bench_gpu3_native.py' in command else foreign).append(app)
        state.update(ours=ours, foreign=foreign, sampled_conflict=bool(ours and foreign))
        stream.write(json.dumps(state)+'\n')
        stream.flush()
        samples += 1
        conflicts += int(state['sampled_conflict'])
        time.sleep(1)
result = dict(start=start, end=time.strftime('%FT%T%z'), screen_pid=a.screen_pid,
    samples=samples, sampled_conflicts=conflicts, interval_seconds=1,
    limitation='Read-only samples supplement existing per-cell pre/end checks; they do not prove sub-second exclusivity.')
(p/'gpu3_monitor_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
