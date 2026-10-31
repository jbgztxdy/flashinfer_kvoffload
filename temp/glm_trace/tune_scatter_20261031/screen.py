"""Paired short-window scatter screen after normal-build cache/Graph correctness passes."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
a = argparse.ArgumentParser()
a.add_argument('--hardware', choices=['gpu3', 'sxm'], required=True)
a.add_argument('--variants', nargs='+', default=['base', 'attention_cache'])
a.add_argument('--positions', type=int, default=256)
a.add_argument('--passes', type=int, default=3)
a.add_argument('--fetch-ctas', type=int, default=24)
a.add_argument('--tag', default='screen256_')
a.add_argument('--candidate', default='attention_cache', choices=['attention_cache', 'owner_cache', 'owner_complement', 'whole_pipe4', 'tail_wave2'])
a = a.parse_args()
from resources import snapshot as sxm_snapshot
from resources_gpu3 import snapshot as gpu3_snapshot
for v in a.variants:
    assert v in ['base', 'attention_cache', 'owner_cache', 'owner_complement', 'whole_pipe4', 'tail_wave2']
gate = p / ('tests_' + a.candidate + '_gpu3.json')
while True:
    t = json.loads(gate.read_text()) if gate.exists() else {}
    if t.get('complete'):
        assert t['exit_code'] == 0
        break
    if 'exit_code' in t:
        raise SystemExit('Normal correctness gate failed; inspect the preserved test record')
    print('WAITING_NORMAL_CORRECTNESS', flush=True)
    time.sleep(20)
gpu3 = a.hardware == 'gpu3'
snapshot = gpu3_snapshot if gpu3 else sxm_snapshot
settings = ['g100_l0', 'g100_l20000', 'g50_l20000'] + ([] if gpu3 else ['nvlink'])
state_file = p / (a.tag + a.hardware + '_state.json')
state = json.loads(state_file.read_text()) if state_file.exists() else []
for v in a.variants:
    while not snapshot()['available']:
        print('WAITING_GPU', snapshot(), flush=True)
        time.sleep(30)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='3' if gpu3 else '0,1',
        CUDA_HOME='/usr/local/cuda-13.1', FLASHINFER_CUDA_ARCH_LIST='8.0', PYTHONPATH=str(root),
        TMPDIR=str(root / 'temp'), MAX_JOBS='4', TUNE_VARIANT=v, TUNE_RUN_TAG=a.tag,
        FLASHINFER_WORKSPACE_BASE=str(p / ('jit_' + v)), FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
    cmd = ['numactl', '-N1' if gpu3 else '-N0', '-m1' if gpu3 else '-m0',
        '/mnt/data/mhy-vllm-venv/bin/python', '-u', str(p / ('bench_gpu3_native.py' if gpu3 else 'bench.py')),
        '--kind', 'lru', '--mtp', '8', '--requests', '128', '--settings', *settings,
        '--passes', str(a.passes), '--positions', str(a.positions), '--fetch-ctas', str(a.fetch_ctas), '--host-split', 'off']
    item = dict(variant=v, start=time.strftime('%FT%T%z'), command=cmd)
    state.append(item)
    print('START_SCREEN', v, a.hardware, flush=True)
    with (p / (a.tag + a.hardware + '_' + v + '.log')).open('a') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        item['pid'] = child.pid
        state_file.write_text(json.dumps(state, indent=2) + '\n')
        item['exit_code'] = child.wait()
    item['end'] = time.strftime('%FT%T%z')
    state_file.write_text(json.dumps(state, indent=2) + '\n')
    if item['exit_code']:
        raise SystemExit(item['exit_code'])
    print('DONE_SCREEN', v, a.hardware, flush=True)
print('SCREEN_COMPLETE', a.hardware, flush=True)
