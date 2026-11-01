"""Count expected cells from frozen plans; keep measured coverage distinct from retention."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read(p/'performance_audit.json');groups=[]
for hardware,cohort in [('gpu3','limited'),('gpu3','native'),('sxm','limited'),('sxm','native')]:
 tag='stream_source_shapes_'+hardware+'_'+cohort+'_';planfile=p/(tag+'plan.json');plan=read(planfile)
 rows=[x for x in a['rows'] if x['cohort']==tag]
 index={(r['variant'],r['kind'],r['mtp'],r['requests'],r['setting'],r['fraction']):r for r in rows}
 expected=[]
 for task in plan['tasks']:
  assert sha(p/(task['variant']+'.cuh'))==plan['candidate_sha256' if task['variant']!='base' else 'base_sha256']
  for setting in task['settings']:
   for fraction in task['fractions'] or [None]:
    key=(task['variant'],task['kind'],task['mtp'],task['requests'],setting,fraction)
    expected.append(key)
    if key in index:
     r=index[key];assert r['positions']==1024 and r['passes']==3 and r['fetch_ctas']==task['fetch_ctas']
 assert len(set(expected))==plan['expected_cells']
 assert set(index)<=set(expected)
 groups.append(dict(hardware=hardware,cohort=cohort,tag=tag,expected_cells=len(expected),
  completed_audited_cells=len(rows),missing_cells=[list(k) for k in expected if k not in index],
  complete=len(rows)==len(expected),plan_sha256=sha(planfile)))
local=[g for g in groups if g['hardware']=='gpu3'];sxm=[g for g in groups if g['hardware']=='sxm']
changed=[r for r in a['pairs'] if r['cohort']=='stream_source_shapes_gpu3_limited_' and r['setting'].startswith('g100')]
normal=[r for r in a['pairs'] if r['cohort']=='stream_source_shapes_gpu3_native_']
negative=[r for r in a['pairs'] if r['reduction']<0]
result=dict(goal_complete=False,candidate_adopted=False,groups=groups,
 gpu3_matrix_complete=all(g['complete'] for g in local),sxm_matrix_complete=all(g['complete'] for g in sxm),
 gpu3_completed_rows=sum(g['completed_audited_cells'] for g in local),
 total_rows=len(a['rows']),total_pairs=len(a['pairs']),pending_references=len(a['pending_references']),incomplete_raw=len(a['incomplete']),
 changed_gpu3_100GB_pairs=len(changed),changed_gpu3_100GB_reduction_range=[min(r['reduction'] for r in changed),max(r['reduction'] for r in changed)] if changed else None,
 negative_pairs=negative,over_one_percent_slow=[r for r in a['pairs'] if r['reduction']<-.01],
 remaining_retention_gates=['Complete missing cells from frozen156matrix plans, including actualNVLink/PCIe','Confirm all large full1024x5pass cohorts in both orders','Preserve and review every over1%negative with full controls','Frozen retention gates, then fresh production root normal compile/115tests/codegen76/actualGraph'],
 audit_sha256=sha(p/'performance_audit.json'))
(p/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ['gpu3_matrix_complete','sxm_matrix_complete','total_rows','total_pairs','pending_references','incomplete_raw']})
