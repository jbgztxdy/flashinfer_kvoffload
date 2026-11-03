"""One focused payload/Graph check and three paired points, under the shared GPU lock."""
import fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources_gpu3 import snapshot,require_exclusive
record=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[])
def save(): (p/'direct_state.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,cmd,env):
    item=dict(label=label,command=cmd,start=time.strftime('%FT%T%z'));record['actions'].append(item);save()
    with (p/(label+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;save();print('START',label,child.pid,flush=True);item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');save();print('END',label,item['exit_code'],flush=True);assert item['exit_code']==0,label
for v in ['square_p']:
    rec=json.loads((p/f'build_{v}.json').read_text());assert len(rec)==2 and all(x['exit_code']==0 for x in rec)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='square_p',FLASHINFER_WORKSPACE_BASE=str(p/'jit_square_p_normal'),FLASHINFER_EXTRA_CUDAFLAGS='')
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
assert snapshot()['available'],snapshot();record['resources_start']=require_exclusive();save()
cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','directpaths',str(p/'test_square_p.py')]
run('direct_targeted_correctness',cmd,env)
record['resources_after_tests']=require_exclusive();save();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
for v in ['base','square_p']:
    cache=p/('jit_'+v) if v!='base' else p.parent/'tune_two_pass_20261102/jit_base'
    bench_env=dict(env,TUNE_VARIANT=v,TUNE_RUN_TAG='direct256_',FLASHINFER_WORKSPACE_BASE=str(cache),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
    cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
    run('direct256_'+v,cmd,bench_env)
record['complete']=True;record['end']=time.strftime('%FT%T%z');save()
