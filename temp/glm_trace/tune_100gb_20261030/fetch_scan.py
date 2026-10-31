"""Probe neighboring fetch-worker counts on the validated whole-row kernel."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
while not snapshot()['available']:time.sleep(30)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU',TUNE_VARIANT='final_neighbor',FLASHINFER_WORKSPACE_BASE=str(p/'jit_final'),TUNE_RUN_TAG='neighbor_')
for fetch in (20,28):
 while not snapshot()['available']:time.sleep(30)
 cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000','g50_l20000','nvlink','--fetch-ctas',str(fetch),'--host-split','off']
 with (p/f'neighbor_f{fetch}.log').open('w') as stream:r=subprocess.run(cmd,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(r.returncode)
 print('DONE fetch',fetch,flush=True)
print('NEIGHBOR_SCAN_COMPLETE',flush=True)
