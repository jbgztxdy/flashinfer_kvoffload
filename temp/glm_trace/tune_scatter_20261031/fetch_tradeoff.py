"""Check whether removing fetch cache scatter can repay fewer reserved fetch CTAs.

This is a new resource-allocation check of the shared-tile mechanism, not a repeat of
the previous unmodified fetch sweep. Compare both variants at every caller parameter.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
p = Path(__file__).resolve().parent
a = argparse.ArgumentParser()
a.add_argument('--candidate', choices=['owner_complement', 'whole_pipe4'], default='owner_complement')
a.add_argument('--fetch-ctas', nargs='+', type=int, default=[16, 8])
a = a.parse_args()
prefix = 'complement' if a.candidate == 'owner_complement' else 'pipe4'
state = []
for fetch in a.fetch_ctas:
    cmd = ['/mnt/data/mhy-vllm-venv/bin/python', '-u', str(p / 'screen.py'),
        '--hardware', 'gpu3', '--candidate', a.candidate, '--variants', a.candidate, 'base',
        '--fetch-ctas', str(fetch), '--tag', prefix + '_f' + str(fetch) + '_']
    item = dict(fetch=fetch, command=cmd, start=time.strftime('%FT%T%z'))
    state.append(item)
    child = subprocess.Popen(cmd, cwd=p.parents[2])
    item['pid'] = child.pid
    (p / ('fetch_tradeoff_' + a.candidate + '_state.json')).write_text(json.dumps(state, indent=2) + '\n')
    item['exit_code'] = child.wait()
    item['end'] = time.strftime('%FT%T%z')
    (p / ('fetch_tradeoff_' + a.candidate + '_state.json')).write_text(json.dumps(state, indent=2) + '\n')
    if item['exit_code']:
        raise SystemExit(item['exit_code'])
