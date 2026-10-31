"""Fresh full-window five-pass confirmation of the runtime-predicate candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=Path(__file__).resolve().parent;root=p.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--hardware',choices=['sxm','gpu3'],required=True)
parser.add_argument('--cohort',choices=['limited','native'],required=True)
a=parser.parse_args()
from resources import snapshot as sxm_snapshot
from resources_gpu3 import snapshot as gpu3_snapshot
gpu3=a.hardware=='gpu3';native=a.cohort=='native'
snapshot=gpu3_snapshot if gpu3 else sxm_snapshot
suffix='_normal' if native else ''
tests=json.loads((p/'tests_runtime_ids_gpu3.json').read_text())
assert tests['complete'] and tests['exit_code']==0
proof=json.loads((p/'compiler_runtime_normal_summary.json').read_text())
assert proof['complete'] and proof['unchanged_original_kernels']==64 and proof['all_new_stack_and_local_zero']
settings=(['native_hbm','pcie'] if gpu3 else ['nvlink','pcie']) if native else ['g100_l0','g100_l20000','g50_l20000']
tag=f'runtime_full_{a.hardware}_{a.cohort}_'
statepath=p/(tag+'state.json')
state=json.loads(statepath.read_text()) if statepath.exists() else []
variants=['runtime_ids','base'] if native else ['base','runtime_ids']
for variant in variants:
    while not snapshot()['available']:
        print('WAITING_GPU',snapshot(),flush=True);time.sleep(10)
    workspace=(p.parent/'tune_heads_20261031' if variant=='base' else p)/('jit_'+variant+suffix)
    assert len(list(workspace.rglob('*host_overlap*.so')))==(2 if native else 1)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3' if gpu3 else '0,1',
        CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
        TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=variant,TUNE_RUN_TAG=tag,
        FLASHINFER_WORKSPACE_BASE=str(workspace),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
    command=['numactl','-N1' if gpu3 else '-N0','-m1' if gpu3 else '-m0',
        '/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/('bench_gpu3_native.py' if gpu3 else 'bench.py')),
        '--kind','lru','--mtp','8','--requests','128','--settings',*settings,
        '--positions','1024','--passes','5','--fetch-ctas','24','--host-split','off']
    item=dict(variant=variant,start=time.strftime('%FT%T%z'),command=command,
        launcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    state.append(item)
    with (p/(tag+variant+'.log')).open('a') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid;statepath.write_text(json.dumps(state,indent=2)+'\n')
        item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');statepath.write_text(json.dumps(state,indent=2)+'\n')
    print('CONFIRM_DONE',variant,item['exit_code'],flush=True)
    if item['exit_code']:raise SystemExit(item['exit_code'])
