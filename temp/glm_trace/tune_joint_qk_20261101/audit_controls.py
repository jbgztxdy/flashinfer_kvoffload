"""Audit5passbrackets independently of original negative matrix samples."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read(p/'performance_audit.json');plan=read(p/'controls_plan.json');rows={r['raw']:r for r in a['rows']};result=[]
for case in plan['cases']:
 evidence=[]
 for phase,v in [('before','base'),('candidate','joint_qk'),('after','base')]:
  tag='joint_qk_control_'+case['name']+'_'+phase+'_'
  f=p/'gpu3/raw'/f"{tag}{v}_staged_f{case['fetch_ctas']}_i64_soff_{case['kind']}_native_hbm_mtp{case['mtp']}_b{case['requests']}.json"
  assert f.exists() and read(f)['complete']
  r=rows[str(f.relative_to(p.parents[2]))];assert r['positions']==1024 and r['passes']==5
  evidence.append(dict(phase=phase,ours_us=r['ours_us'],raw=r['raw'],sha256=sha(f),pass_mean_us=r['pass_mean_us']))
 before,new,after=[x['ours_us'] for x in evidence];base=(before+after)/2
 result.append(dict(case=case['name'],before_us=before,candidate_us=new,after_us=after,reduction_vs_before=1-new/before,reduction_vs_after=1-new/after,
  reduction_vs_baseline_mean=1-new/base,baseline_endpoint_change=after/before-1,
  over_one_percent_slow=(1-new/before)<-.01 and (1-new/after)<-.01,
  original_samples_preserved=True,evidence=evidence))
out=dict(complete=True,candidate_adopted=False,goal_status='active',cases=result,
 all_within_one_percent_each_endpoint=all(x['reduction_vs_before']>=-.01 and x['reduction_vs_after']>=-.01 for x in result),
 plan_sha256=sha(p/'controls_plan.json'),performance_audit_sha256=sha(p/'performance_audit.json'),
 limitation='Brackets establish repeatability of these two independent-allocation measurements, not a causal attribution. Original slow cells stay in performance_audit.json.')
(p/'controls_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print([{k:r[k] for k in ['case','before_us','candidate_us','after_us','reduction_vs_baseline_mean','baseline_endpoint_change']} for r in result])
