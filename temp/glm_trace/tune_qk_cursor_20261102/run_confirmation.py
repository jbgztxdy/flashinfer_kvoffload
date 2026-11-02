"""Focused confirmation, no matrix or exhaustive suite."""
import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
assert json.loads((p/'focused_state.json').read_text())['complete']
record=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[])
def save():(p/'confirmation_state.json').write_text(json.dumps(record,indent=2)+'\n')
for native,positions,tag,settings in [(False,1024,'reverse1024_',['g100_l0','g100_l20000','g50_l20000']),(True,256,'native256_',['native_hbm','pcie'])]:
 for v in ['qk_cursor','base']:
  env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=v,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+('_normal' if native else ''))),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
  cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench_gpu3_native.py'),'--kind','lru','--mtp','8','--requests','128','--settings',*settings,'--passes','3','--positions',str(positions),'--fetch-ctas','24','--host-split','off']
  label=tag+v;item=dict(label=label,command=cmd,start=time.strftime('%FT%T%z'));record['actions'].append(item)
  with (p/(label+'.log')).open('w') as log:
   child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;save();print('START',label,child.pid,flush=True);item['exit_code']=child.wait()
  item['end']=time.strftime('%FT%T%z');save();print('END',label,item['exit_code'],flush=True);assert item['exit_code']==0
record.update(complete=True,end=time.strftime('%FT%T%z'));save()
