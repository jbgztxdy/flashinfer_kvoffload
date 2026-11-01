"""Validate restored-fetch control, then run one fixed same-parameter cohort."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];py='/mnt/data/mhy-vllm-venv/bin/python'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
read=lambda f:json.loads(f.read_text())
variants=['no_copy_pe_fixed','half_ring'];out=p/'validation_fixed_state.json'
state=dict(start=time.strftime('%FT%T%z'),pid=os.getpid(),phase='waiting_fixed_build_tests',complete=False,candidate_adopted=False,goal_complete=False,steps=[])
def save():
    t=out.with_suffix('.json.new');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out)
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
        done=True
        for v in variants:
            for prefix in ['tests','stress']:
                f=p/(prefix+'_'+v+'_gpu3.json');d=read(f) if f.exists() else {}
                for k in ['compile_exit_code','exit_code']:
                    if d.get(k) not in (None,0):raise RuntimeError('Failed original '+str(f))
                done &= d.get('complete',False)
            f=p/('compile_'+v+'_emu.log');done &= f.exists() and ('COMPILED '+v) in f.read_text()
        if done:break
        time.sleep(10)
    state['phase']='fixed_codegen_graph_sanitizers';save()
    for normal in [False,True]:
        run([py,str(p/'compiler.py'),'--variants','no_copy_pe_fixed']+(['--normal'] if normal else []),'compiler_fixed'+('_normal' if normal else '')+'.log')
    for v in variants:
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',CUDA_HOME='/usr/local/cuda-13.1',
            FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='4',
            TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
        prefix=['numactl','-N1','-m1'];probe=[py,'-u',str(p/('probe_'+v+'.py'))]
        if v=='no_copy_pe_fixed':run(prefix+probe,'probe_'+v+'.log',env)
        tools=['synccheck','racecheck','memcheck'] if v=='no_copy_pe_fixed' else ['memcheck']
        for kind in tools:
            cmd=prefix+['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',kind,'--launch-timeout','0','--error-exitcode','86',
                '--kernel-name','regex='+('BatchMLAHalfRingKernel' if v=='half_ring' else 'BatchMLANoCopyPEKernel'),'--launch-count','1']
            if kind=='racecheck':cmd+=['--racecheck-num-workers','8']
            if kind=='memcheck':cmd+=['--report-api-errors','no']
            log=kind+'_'+v+'.log';run(cmd+probe,log,env)
            text=(p/log).read_text();assert 'ERROR SUMMARY: 0 errors' in text or (kind=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text),log
    plan=dict(positions=256,passes=3,mtp=8,requests=128,fetch_ctas=24,host_split='off',settings=['g100_l0','g100_l20000','g50_l20000'],
        order=['base',*variants],headers={v:sha(p/(v+'.cuh')) for v in ['base',*variants]},
        comparison_base='Frozen cachedPEa436, equivalent parent of productionJQ7193 plus pending itemQPEreuse',
        limitation='Short screen only. Control restores original147968B/two64row fetch buffers; half-ring changes attention storage/layout/loader. No fulltrace or adoption.')
    (p/'short_fixed_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state['phase']='fixed_short_threeway_screen';save()
    run([py,'-u',str(p/'screen_fixed.py'),'--hardware','gpu3','--candidate','half_ring','--variants',*plan['order'],
        '--requests','128','--mtp','8','--fetch-ctas','24','--host-split','off','--settings',*plan['settings'],'--tag','half_fixed256_f24_'],'screen_fixed_gpu3.log')
    run([py,str(p/'audit_performance.py')],'audit_fixed_performance.log')
    state.update(phase='fixed_short_screen_complete',complete=True,end=time.strftime('%FT%T%z'));save()
    print('FIXED_SHORT_COMPARISON_COMPLETE_NO_INSTALL',flush=True)
except Exception as exc:
    state.update(phase='failed',error=str(exc),end=time.strftime('%FT%T%z'));save();raise
