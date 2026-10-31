"""Serialize an instrumented 128-row phase profile on the final whole-row path."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU',TUNE_VARIANT='profile_final',TUNE_RUN_TAG='profile_final_',FLASHINFER_WORKSPACE_BASE=str(p/'jit_profile_final'))
while True:
 s=snapshot()
 if s['available']:break
 (p/'waiting_phase_final.json').write_text(json.dumps(s,indent=2)+'\n');time.sleep(30)
cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l20000','--passes','1','--fetch-ctas','24','--host-split','off']
with (p/'phase_final.log').open('w') as stream:r=subprocess.run(cmd,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
raise SystemExit(r.returncode)
