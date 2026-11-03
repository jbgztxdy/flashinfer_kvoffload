"""Actual single-kernel probe plus three reverse short pairs; no broad repeat gates."""
import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'performance_audit.json').read_text())
assert len(a['pairs'])==3
assert all(x['reduction']>0 for x in a['pairs'] if x['setting'].startswith('g100'))
assert json.loads((p/'state.json').read_text())['complete']
r=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[])
def save():(p/'confirmation_state.json').write_text(json.dumps(r,indent=2)+'\n')
def run(label,cmd,env):
 item=dict(label=label,start=time.strftime('%FT%T%z'),command=cmd);r['actions'].append(item);save()
 with (p/(label+'.log')).open('w') as f:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  item['pid']=child.pid;save();print('START',label,child.pid,flush=True);item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');save();print('END',label,item['exit_code'],flush=True)
 assert item['exit_code']==0,label
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='pv_feature',FLASHINFER_WORKSPACE_BASE=str(p/'jit_pv_feature_normal'),FLASHINFER_EXTRA_CUDAFLAGS='')
prefix=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u']
run('actual_dispatch',prefix+[str(p/'probe.py')],env)
for normal,settings in [(False,['g100_l0','g100_l20000']),(True,['native_hbm'])]:
 for v in ['pv_feature','base']:
  cache=p/('jit_'+v+('_normal' if normal else '')) if v!='base' else p.parent/'tune_two_pass_20261102'/('jit_base'+('_normal' if normal else ''))
  tag='native256_' if normal else 'reverse256_'
  bench_env=dict(env,TUNE_VARIANT=v,TUNE_RUN_TAG=tag,FLASHINFER_WORKSPACE_BASE=str(cache),FLASHINFER_EXTRA_CUDAFLAGS='' if normal else '-DFLASHINFER_FAR_EMU')
  cmd=prefix+[str(p/'bench_gpu3_native.py'),'--kind','lru','--mtp','8','--requests','128','--settings',*settings,'--passes','3','--positions','256','--fetch-ctas','24','--host-split','off']
  run(tag+v,cmd,bench_env)
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
