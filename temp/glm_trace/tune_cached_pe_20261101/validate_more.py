"""Run the frozen regression scope, preserving the original JQ producer's priority."""
import json
import os
from pathlib import Path
import subprocess
import time

p=Path(__file__).resolve().parent
root=p.parents[2]
py='/mnt/data/mhy-vllm-venv/bin/python'
read=lambda f:json.loads(f.read_text())
out=p/'broader_validation_state.json'
state=dict(start=time.strftime('%FT%T%z'),pid=os.getpid(),phase='gpu3_matrix',complete=False,
    candidate_adopted=False,goal_complete=False,steps=[])
def save():
    tmp=out.with_suffix('.json.new')
    tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
def run(cmd,name):
    item=dict(command=cmd,log=name,start=time.strftime('%FT%T%z'))
    state['steps'].append(item)
    with (p/name).open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        item['pid']=child.pid;save();item['exit_code']=child.wait()
    item['end']=time.strftime('%FT%T%z');save()
    if item['exit_code']:raise RuntimeError('Failed '+name)
def refresh():
    for script in ['audit_performance.py','audit_coverage.py','report.py']:
        run([py,str(p/script)],script.removesuffix('.py')+'.log')
def gated(pairs):
    for cohort in ['cachedreverse1024_f24_','cachedforward1024_f24_']:
        rows=[r for r in pairs if r['cohort']==cohort]
        assert len(rows)==3 and all(r['reduction']>0 for r in rows)
    rows=[r for r in pairs if r['cohort']=='cachednative1024_f24_']
    assert len(rows)==2 and all(r['reduction']>-.01 for r in rows)
save()
try:
    assert read(p/'extension_state.json')['complete']
    gated(read(p/'performance_audit.json')['pairs'])
    for cohort in ['limited','native']:
        run([py,'-u',str(p/'validate_shapes.py'),'--cohort',cohort], 'matrix_gpu3_'+cohort+'.log')
    refresh()
    state['gpu3_matrix_complete']=True
    state['phase']='waiting_original_JQ_forward'
    save()
    jq=p.parent/'tune_joint_qk_20261101'
    while True:
        missing=[]
        for setting in ['g100_l0','g100_l20000','g50_l20000']:
            f=jq/'raw'/f'compositionsxmforward1024_f24_joint_qk_staged_f24_i64_soff_lru_{setting}_mtp8_b128.json'
            if not f.exists() or not read(f).get('complete'):missing.append(setting)
        state['missing_original_JQ']=missing
        state['last_observation']=time.strftime('%FT%T%z')
        if not missing:
            save();break
        ps=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        live=[s.strip() for s in ps if '/python' in s and
            ('tune_joint_qk_20261101/confirm_sxm.py' in s or 'tune_joint_qk_20261101/complete_retention.py' in s)]
        state['live_original_producers']=live
        save()
        if not live:raise RuntimeError('Original JQ producer is missing while its required raw remains incomplete; inspect original state, do not duplicate it')
        print('WAIT_ORIGINAL_JQ_FORWARD',missing,live,flush=True)
        time.sleep(30)
    state['phase']='sxm_full_confirmation';save()
    tasks=[dict(tag='cachedsxmreverse1024_f24_',variants=['cached_pe','base'],native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
           dict(tag='cachedsxmforward1024_f24_',variants=['base','cached_pe'],native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
           dict(tag='cachedsxmnative1024_f24_',variants=['base','cached_pe'],native=True,settings=['nvlink','pcie'])]
    (p/'sxm_confirmation_plan.json').write_text(json.dumps(dict(positions=1024,passes=5,mtp=8,requests=128,fetch_ctas=24,host_split='off',tasks=tasks),indent=2)+'\n')
    for task in tasks:
        cmd=[py,'-u',str(p/'screen.py'),'--hardware','sxm','--candidate','cached_pe',
            '--variants',*task['variants'],'--positions','1024','--passes','5','--requests','128',
            '--mtp','8','--fetch-ctas','24','--host-split','off','--tag',task['tag'],'--settings',*task['settings']]
        if task['native']:cmd+=['--native']
        run(cmd,task['tag']+'screen.log')
    refresh()
    pairs=read(p/'performance_audit.json')['pairs']
    for cohort in ['cachedsxmreverse1024_f24_','cachedsxmforward1024_f24_']:
        rows=[r for r in pairs if r['cohort']==cohort]
        assert len(rows)==3 and all(r['reduction']>0 for r in rows if r['setting'].startswith('g100'))
        assert all(r['reduction']>-.01 for r in rows)
    rows=[r for r in pairs if r['cohort']=='cachedsxmnative1024_f24_']
    assert len(rows)==2 and all(r['reduction']>-.01 for r in rows)
    state['sxm_confirmation_gate_passed']=True
    state['phase']='sxm_matrix';save()
    for cohort in ['limited','native']:
        run([py,'-u',str(p/'validate_sxm.py'),'--cohort',cohort], 'matrix_sxm_'+cohort+'.log')
    refresh()
    state.update(phase='regression_measurements_complete',complete=True,end=time.strftime('%FT%T%z'))
    save()
    print('REGRESSION_MEASUREMENTS_COMPLETE_NO_INSTALL',flush=True)
except Exception as exc:
    state.update(phase='failed',error=str(exc),end=time.strftime('%FT%T%z'))
    save()
    raise
