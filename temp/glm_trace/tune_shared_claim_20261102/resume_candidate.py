"""Only the missing candidate timing, with a fresh resource preflight; no hidden waiter."""
import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources import snapshot
state=snapshot()
(p/'resume_preflight.json').write_text(json.dumps(state,indent=2)+'\n')
assert state['available'],state
validation=json.loads((p/'focused_sxm_state.json').read_text())['actions'][0]
assert validation['label']=='sxm_targeted_correctness' and validation['exit_code']==0
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='shared_claim',FLASHINFER_WORKSPACE_BASE=str(p/'jit_shared_claim'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU',TUNE_RUN_TAG='sxm256_')
cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_sxm.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
record=dict(start=time.strftime('%FT%T%z'),command=cmd)
with (p/'resume_candidate.log').open('w') as log:
 child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 record['pid']=child.pid;(p/'resume_state.json').write_text(json.dumps(record,indent=2)+'\n')
 record['exit_code']=child.wait()
record['end']=time.strftime('%FT%T%z');(p/'resume_state.json').write_text(json.dumps(record,indent=2)+'\n')
assert record['exit_code']==0
