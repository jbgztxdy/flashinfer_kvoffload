"""Focused current-source route choice; no kernel changes, exhaustive tests or matrix."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_qk_cursor_20261102';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text())
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==m['header_sha256']
state=dict(start=time.strftime('%FT%T%z'),complete=False,actions=[])
plans=[]
# Same settings per route, only staging toggles. Recommended staged f24 for large rows.
for r in [16,128]:
 for native in [True,False]:
  for route in (['inplace','staged'] if r==16 else ['staged','inplace']):
   plans.append(dict(kind='lru',requests=r,native=native,route=route,positions=1024,passes=3,settings=['nvlink'] if native else ['g100_l0','g100_l20000'],fetch=24 if r==128 else 16,split='off' if r==128 else 'on'))
# Four original snapshots, include matching zero-far self-control and moderate far ratios.
for native in [True,False]:
 for route in ['staged','inplace']:
  plans.append(dict(kind='nocache',requests=16,native=native,route=route,positions=1024,passes=3,settings=['nvlink'] if native else ['g100_l0','g100_l20000'],fetch=16,split='on',fractions=[0,.1,.3]))
(p/'plan.json').write_text(json.dumps(plans,indent=2)+'\n')
for plan in plans:
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='current',TUNE_RUN_TAG='matched_',FLASHINFER_WORKSPACE_BASE=str(old/('jit_qk_cursor_normal' if plan['native'] else 'jit_qk_cursor')),FLASHINFER_EXTRA_CUDAFLAGS='' if plan['native'] else '-DFLASHINFER_FAR_EMU')
 label=f"{plan['kind']}_r{plan['requests']}_{'normal' if plan['native'] else 'emu'}_{plan['route']}"
 cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind',plan['kind'],'--mtp','8','--requests',str(plan['requests']),'--settings',*plan['settings'],'--passes',str(plan['passes']),'--positions',str(plan['positions']),'--fetch-ctas',str(plan['fetch']),'--host-split',plan['split'],'--route',plan['route']]
 if 'fractions' in plan:cmd+=['--fractions',*[str(x) for x in plan['fractions']]]
 item=dict(plan=plan,command=cmd,label=label,start=time.strftime('%FT%T%z'));state['actions'].append(item)
 with (p/(label+'.log')).open('w') as log:
  child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;(p/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('START',label,child.pid,flush=True);item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');(p/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('END',label,item['exit_code'],flush=True)
 assert item['exit_code']==0
assert all(sha(root/x['path'])==x['sha256'] for x in m['reused_libraries'])
state.update(complete=True,end=time.strftime('%FT%T%z'));(p/'state.json').write_text(json.dumps(state,indent=2)+'\n')
