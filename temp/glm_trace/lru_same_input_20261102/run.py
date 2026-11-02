import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_qk_cursor_20261102'
state=dict(complete=False,start=time.strftime('%FT%T%z'),actions=[])
for native in [True,False]:
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='current',TUNE_RUN_TAG='sameinput_',TUNE_DISABLE_LRU='1',FLASHINFER_WORKSPACE_BASE=str(old/('jit_qk_cursor_normal' if native else 'jit_qk_cursor')),FLASHINFER_EXTRA_CUDAFLAGS='' if native else '-DFLASHINFER_FAR_EMU')
 label='capacity0_'+('normal' if native else 'emu');cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--settings','nvlink' if native else 'g100_l0','--passes','3','--positions','1024','--fetch-ctas','24','--host-split','off','--route','staged']
 item=dict(label=label,command=cmd,start=time.strftime('%FT%T%z'));state['actions'].append(item)
 with (p/(label+'.log')).open('w') as log:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;(p/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('START',label,child.pid,flush=True);item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');(p/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('END',label,item['exit_code'],flush=True);assert item['exit_code']==0
state.update(complete=True,end=time.strftime('%FT%T%z'));(p/'state.json').write_text(json.dumps(state,indent=2)+'\n')
