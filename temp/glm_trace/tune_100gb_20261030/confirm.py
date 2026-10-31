"""Five-pass confirmations after validation/tests, with old code measured first."""
import json,os,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
previous=json.loads((p/'signal_tests.json').read_text())['child_pid']
while True:
 try:os.kill(previous,0)
 except ProcessLookupError:break
 time.sleep(30)
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU',TUNE_RUN_TAG='confirm_')
for m,b in ((8,128),(8,64),(1,512)):
 for v,auto in [('base',False),('base',True),('final',True)]:
  while not snapshot()['available']:time.sleep(30)
  current=dict(env,TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v)))
  cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp',str(m),'--requests',str(b),'--settings','g100_l0','g100_l20000','g50_l20000','nvlink','--passes','5']
  if auto:cmd.append('--auto')
  print('START',v,auto,m,b,time.strftime('%FT%T%z'),flush=True)
  with (p/f'confirm_{v}_auto{auto}_m{m}_b{b}.log').open('w') as stream:r=subprocess.run(cmd,env=current,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
  if r.returncode:raise SystemExit(r.returncode)
  print('DONE',v,auto,m,b,flush=True)
print('CONFIRMATION_COMPLETE',flush=True)
