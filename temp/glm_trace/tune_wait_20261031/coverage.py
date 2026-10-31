"""Count planned regression cells from audited raw; never infer completion from state."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
a = json.loads((p/'performance_audit.json').read_text())
coverage={}
for f in sorted(set(p.glob('ids_matrix_*_plan.json')) | set(p.glob('cached_ids_matrix_*_plan.json'))):
    plan=json.loads(f.read_text())
    expected=set()
    for task in plan['tasks']:
        for setting in task['settings']:
            for fraction in task['fractions'] if task['fractions'] is not None else [None]:
                expected.add((task['variant'],task['kind'],task['mtp'],task['requests'],task['fetch_ctas'],setting,fraction))
    assert len(expected)==plan['expected_cells']
    actual={(r['variant'],r['kind'],r['mtp'],r['requests'],r['fetch_ctas'],r['setting'],r['fraction'])
        for r in a['rows'] if r['cohort']==plan['tag']}
    assert actual<=expected
    assert all(r['positions']==1024 and r['passes']==3 for r in a['rows'] if r['cohort']==plan['tag'])
    pairs=[r for r in a['pairs'] if r['cohort']==plan['tag']]
    coverage[plan['tag']]=dict(actual=len(actual),expected=len(expected),complete=actual==expected,
        pairs=len(pairs),minimum_reduction=min((r['reduction'] for r in pairs),default=None),
        slower_than_1pct=[r for r in pairs if r['reduction']<-.01])
result=dict(goal_complete=False,candidate_adopted=False,coverage=coverage,
    cached_planned_complete=bool([x for x in coverage if x.startswith('cached_ids_')]) and all(r['complete'] for k,r in coverage.items() if k.startswith('cached_ids_')),
    all_planned_complete=bool(coverage) and all(r['complete'] for r in coverage.values()))
(p/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{q:v for q,v in r.items() if q!='slower_than_1pct'} for k,r in coverage.items()},indent=2))
