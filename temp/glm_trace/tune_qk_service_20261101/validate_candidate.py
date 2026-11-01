"""Gate one fixed QK/PV role comparison on actual tests, codegen and sanitizers."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];py='/mnt/data/mhy-vllm-venv/bin/python'
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
out=p/'validation_state.json';v='qks'
state=dict(pid=os.getpid(),start=time.strftime('%FT%T%z'),phase='waiting_build_test_stress',complete=False,candidate_adopted=False,goal_complete=False,steps=[])
def save():
    t=out.with_suffix('.json.new');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out)
def run(cmd,log,env=None):
    item=dict(command=cmd,log=log,start=time.strftime('%FT%T%z'));state['steps'].append(item)
    with (p/log).open('a') as f:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save();item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');save()
    if item['exit_code']:raise RuntimeError('Failed '+log)
save()
try:
    while True:
        done=True
        for prefix in ['tests','stress']:
            f=p/(prefix+'_'+v+'_gpu3.json');d=read(f) if f.exists() else {}
            for key in ['compile_exit_code','exit_code']:
                if d.get(key) not in (None,0):raise RuntimeError('Failed '+str(f))
            done &= d.get('complete',False)
        emu=(p/('compile_'+v+'_emu.log')).read_text()
        if 'Traceback (most recent call last)' in emu:raise RuntimeError('EMU build failed; original producer must be inspected')
        done &= 'COMPILED '+v in emu
        if done:break
        time.sleep(10)
    state['phase']='codegen_graph_sanitizers';save()
    for normal in [False,True]:run([py,str(p/'compiler.py')]+(['--normal'] if normal else []),'compiler'+('_normal' if normal else '')+'.log')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',
        FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',TUNE_VARIANT=v,
        FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
    prefix=['numactl','-N1','-m1'];probe=[py,'-u',str(p/('probe_'+v+'.py'))]
    run(prefix+probe,'probe_'+v+'.log',env)
    for kind in ['synccheck','racecheck','memcheck']:
        cmd=prefix+['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',kind,'--launch-timeout','0','--error-exitcode','86',
            '--kernel-name','regex=BatchMLAQKSServiceKernel','--launch-count','1']
        if kind=='racecheck':cmd+=['--racecheck-num-workers','8']
        if kind=='memcheck':cmd+=['--report-api-errors','no']
        log=kind+'_'+v+'.log';run(cmd+probe,log,env)
        text=(p/log).read_text();assert 'ERROR SUMMARY: 0 errors' in text or (kind=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text)
    plan=dict(variants=['base','control_qp',v],positions=256,passes=3,mtp=8,requests=128,fetch_ctas=24,host_split='off',
        settings=['g100_l0','g100_l20000','g50_l20000'],headers={n:sha(p/(n+'.cuh')) for n in ['base','control_qp',v]},
        baseline='Frozen cachedPEa436,notproductionJQ7193',claim='shortscreenonly; no fulltrace/native/adoption proof')
    (p/'short_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state['phase']='fixed_short_screen';save()
    run([py,'-u',str(p/'screen.py'),'--hardware','gpu3','--candidate',v,'--variants','base','control_qp',v,'--settings',*plan['settings'],
        '--requests','128','--mtp','8','--fetch-ctas','24','--host-split','off','--tag','qks256_f24_'],'screen_gpu3.log')
    run([py,str(p/'audit_performance.py')],'audit_performance.log')
    state.update(phase='short_screen_complete',complete=True,end=time.strftime('%FT%T%z'));save();print('SHORT_SCREEN_COMPLETE_NO_INSTALL',flush=True)
except Exception as exc:
    state.update(phase='failed',error=str(exc),end=time.strftime('%FT%T%z'));save();raise
