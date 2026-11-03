"""Fresh same-hardware paired screen; old SXM baseline cannot be used for GPU3."""
import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources_gpu3 import snapshot
record=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[],resources_start=snapshot())
assert record['resources_start']['available'],record['resources_start']
def save():(p/'gpu3_20261103_state.json').write_text(json.dumps(record,indent=2)+'\n')
save()
for v in ['base','shared_claim']:
 cache=p/('jit_'+v) if v!='base' else p.parent/'tune_two_pass_20261102/jit_base'
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(cache),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU',TUNE_RUN_TAG='gpu3_20261103_')
 cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
 item=dict(variant=v,command=cmd,start=time.strftime('%FT%T%z'));record['actions'].append(item);save()
 with (p/f'gpu3_20261103_{v}.log').open('w') as log:
  child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
  print('START',v,child.pid,flush=True);item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');save();print('END',v,item['exit_code'],flush=True);assert item['exit_code']==0
record['complete']=True;record['end']=time.strftime('%FT%T%z');record['resources_end']=snapshot();save()
