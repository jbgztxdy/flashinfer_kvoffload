"""Preserve every >1% local negative and measure full before/candidate/after controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

p=Path(__file__).resolve().parent
root=p.parents[2]
py='/mnt/data/mhy-vllm-venv/bin/python'
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
out=p/'controls_state.json'
state=dict(start=time.strftime('%FT%T%z'),phase='wait_local_matrix_and_original_root_validation',complete=False,steps=[])
def save():
    tmp=out.with_suffix('.json.new');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
save()
try:
    jq=p.parent/'tune_joint_qk_20261101'
    while True:
        more=read(p/'broader_validation_state.json')
        old=read(jq/'completion_state.json')
        if more['phase']=='failed' or old['phase']=='failed':
            raise RuntimeError('Original broader validation/retention failed; inspect its preserved state')
        if more.get('gpu3_matrix_complete') and old.get('complete'):break
        ps=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        live=[s.strip() for s in ps if '/python' in s and
              ('tune_cached_pe_20261101/validate_more.py' in s or 'tune_joint_qk_20261101/complete_retention.py' in s)]
        assert live,'Expected original producers missing; do not duplicate'
        state.update(last_observation=time.strftime('%FT%T%z'),live_original_producers=live);save()
        time.sleep(15)
    a=read(p/'performance_audit.json')
    points=[r for r in a['pairs'] if r['cohort'].startswith('cached_pe_shapes_gpu3_') and r['reduction']<-.01]
    assert len(points)==2 and {(r['kind'],r['mtp'],r['requests'],r['setting']) for r in points}=={
        ('lru',1,1,'native_hbm'),('lru',1,256,'native_hbm')}
    plan=dict(recorded_at=time.strftime('%FT%T%z'),positions=1024,passes=5,host_split='off',
        points=points,order=['base_before','cached_pe','base_after'],
        original_audit_sha256=sha(p/'performance_audit.json'),
        reason='Two unchanged-path negatives>1%; original samples remain. Same source/config, full5coldpasses each side. No numerical or retention gate weakened.')
    (p/'controls_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state['phase']='full_brackets';save()
    controls=[]
    for point in points:
        raws=[]
        for side,variant in [('before','base'),('candidate','cached_pe'),('after','base')]:
            tag=f"unchanged_m{point['mtp']}_r{point['requests']}_{side}_"
            cmd=[py,'-u',str(p/'screen.py'),'--hardware','gpu3','--candidate','cached_pe','--native',
                '--variants',variant,'--positions','1024','--passes','5','--requests',str(point['requests']),
                '--mtp',str(point['mtp']),'--fetch-ctas',str(point['fetch_ctas']),'--host-split','off',
                '--settings',point['setting'],'--tag',tag]
            step=dict(command=cmd,start=time.strftime('%FT%T%z'));state['steps'].append(step)
            with (p/(tag+'screen.log')).open('a') as log:
                child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                step['pid']=child.pid;save();step['exit_code']=child.wait()
            step['end']=time.strftime('%FT%T%z');save()
            if step['exit_code']:raise RuntimeError('Failed bracket '+tag)
            f=p/'gpu3/raw'/f"{tag}{variant}_staged_f{point['fetch_ctas']}_i64_soff_lru_{point['setting']}_mtp{point['mtp']}_b{point['requests']}.json"
            d=read(f);assert d['complete'] and d['configuration']['passes']==5
            assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
            t=d['cases'][0]['timings']['ours']
            raws.append(dict(side=side,variant=variant,raw=str(f.relative_to(root)),sha256=sha(f),
                mean_us=t['mean_us'],pass_mean_us=t['pass_mean_us'],header_sha256=d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']))
        baseline=(raws[0]['mean_us']+raws[2]['mean_us'])/2
        reduction=1-raws[1]['mean_us']/baseline
        controls.append(dict(original_point=point,raws=raws,baseline_mean_us=baseline,reduction=reduction,
            over_one_percent_negative_reproduced=reduction<-.01))
        (p/'controls_audit.json').write_text(json.dumps(dict(complete=False,controls=controls),indent=2)+'\n')
    result=dict(complete=True,goal_complete=False,candidate_adopted=False,controls=controls,
        all_original_negatives_preserved=True,
        no_over_one_percent_negative_reproduced=all(not r['over_one_percent_negative_reproduced'] for r in controls),
        limit='Brackets quantify reproducibility. Unchanged machine words and non-reproduction do not prove a unique cause or promise every independent allocation is faster.')
    (p/'controls_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    state.update(phase='complete',complete=True,end=time.strftime('%FT%T%z'));save()
    print('BRACKET_CONTROLS_COMPLETE',[(r['original_point']['requests'],r['reduction']) for r in controls],flush=True)
except Exception as exc:
    state.update(phase='failed',error=str(exc),end=time.strftime('%FT%T%z'));save();raise
