"""Finish original producers, then gate one fixed half-ring/control comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=Path(__file__).resolve().parent;root=p.parents[2]
py='/mnt/data/mhy-vllm-venv/bin/python'
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
variants=['no_copy_pe','half_ring']
out=p/'validation_state.json'
state=dict(start=time.strftime('%FT%T%z'),pid=os.getpid(),phase='waiting_original_build_test_stress',
    complete=False,candidate_adopted=False,goal_complete=False,steps=[])
def save():
    tmp=out.with_suffix('.json.new');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
def run(cmd,name,env=None):
    item=dict(command=cmd,log=name,start=time.strftime('%FT%T%z'));state['steps'].append(item)
    with (p/name).open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid;save();item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');save()
    if item['exit_code']:raise RuntimeError('Failed '+name)
save()
try:
    while True:
        all_done=True
        for v in variants:
            t=read(p/('tests_'+v+'_gpu3.json'))
            sf=p/('stress_'+v+'_gpu3.json');st=read(sf) if sf.exists() else {}
            for d in [t,st]:
                for key in ['compile_exit_code','exit_code']:
                    if d.get(key) not in (None,0):raise RuntimeError('Original build/test failed: '+v)
            all_done &= t.get('complete',False) and st.get('complete',False) and ('COMPILED '+v in (p/('compile_'+v+'_emu.log')).read_text())
        if all_done:break
        time.sleep(10)
    state['phase']='codegen_words_graph_sanitizers';save()
    for normal in [False,True]:
        run([py,str(p/'compiler.py')]+(['--normal'] if normal else []),'compiler'+('_normal' if normal else '')+'.log')
    for v in variants:
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',
            CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),
            TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=v,
            FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
        prefix=['numactl','-N1','-m1']
        run(prefix+[py,'-u',str(p/('probe_'+v+'.py'))],'probe_'+v+'.log',env)
        for kind in ['synccheck','racecheck']:
            cmd=prefix+['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',kind,'--launch-timeout','0',
                '--error-exitcode','86','--kernel-name','regex='+('BatchMLAHalfRingKernel' if v=='half_ring' else 'BatchMLANoCopyPEKernel'),'--launch-count','1']
            if kind=='racecheck':cmd+=['--racecheck-num-workers','8']
            run(cmd+[py,'-u',str(p/('probe_'+v+'.py'))],kind+'_'+v+'.log',env)
    plan=dict(positions=256,passes=3,mtp=8,requests=128,fetch_ctas=24,host_split='off',
        settings=['g100_l0','g100_l20000','g50_l20000'],order=['base','no_copy_pe','half_ring'],
        headers={v:sha(p/(v+'.cuh')) for v in ['base',*variants]},
        comparison_base='Frozen cachedPEa436; currentproductionJQ7193 unchanged',
        limitation='Short screen only,not full trace or adoption. Two-stage control isolates unused shared QPE removal; half-ring also changes P layout/loader,not an isolated latency attribution.')
    (p/'short_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state['phase']='fixed_short_threeway_screen';save()
    run([py,'-u',str(p/'screen.py'),'--hardware','gpu3','--candidate','half_ring',
        '--variants',*plan['order'],'--requests','128','--mtp','8','--fetch-ctas','24',
        '--host-split','off','--settings',*plan['settings'],'--tag','half256_f24_'],'screen_gpu3.log')
    run([py,str(p/'audit_performance.py')],'audit_performance.log')
    state.update(phase='short_screen_complete',complete=True,end=time.strftime('%FT%T%z'));save()
    print('SHORT_COMPARISON_COMPLETE_NO_INSTALL',flush=True)
except Exception as exc:
    state.update(phase='failed',error=str(exc),end=time.strftime('%FT%T%z'));save();raise
