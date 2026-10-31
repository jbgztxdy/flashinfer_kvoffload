"""Audit the shared-allocation diagnosis without treating it as matrix replacement."""
import hashlib
import json
import statistics
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((p/'allocation_probe.json').read_text())
assert d['complete'] and d['diagnostic'] and d['shared_query_indices_cache_workspace_output_and_KV']
assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
assert d['source_sha256']==sha(p/'allocation_probe.py')
assert len(d['cases'])==2
summary=[]
for c in d['cases']:
    assert len(c['dispatch'])==3 and len({r['launch']['function'] for r in c['dispatch'].values()})==3
    assert all(r['attention_kernel_nodes']==128 for r in c['dispatch'].values())
    means={};misses={}
    for variant in ['base','cached_ids','runtime_ids']:
        rows=[r for r in c['records'] if r['variant']==variant]
        assert len(rows)==5
        for r in rows:
            assert len(r['per_step_us'])==len(r['unique_misses'])==128
            assert abs(statistics.mean(r['per_step_us'])-r['mean_us'])<1e-8
            assert r['max_abs_error']<.01 and r['cache_payload_checked_rows']==128*4096
        means[variant]=statistics.mean(r['mean_us'] for r in rows)
        misses[variant]=statistics.mean(x for r in rows for x in r['unique_misses'])
    summary.append(dict(setting=c['setting'],mean_us=means,mean_unique_misses=misses,
        reductions={v:1-means[v]/means['base'] for v in ['cached_ids','runtime_ids']}))
snap=p/'source_snapshots'
for f in [p/'allocation_probe.py',p.parent/'tune_wait_20261031/bench.py']:
    target=snap/(sha(f)+f.suffix)
    if target.exists():assert target.read_bytes()==f.read_bytes()
    else:target.write_bytes(f.read_bytes())
result=dict(complete=True,diagnostic=True,candidate_adopted=False,goal_complete=False,
    actual_distinct_functions_and_equal_launch_shape=True,passes_each=5,positions=1024,
    checks='All30passes:sampledoutputs,cachebijection,andactual524288cachedKVrows',
    raw_sha256=sha(p/'allocation_probe.json'),summary=summary,
    limitation='Sameallocationdiagnosisisnotareplacementfortheindependentprocessmatrix;allnegativecellsremain.Causationforcross-runvariationsnotestablished.')
(p/'allocation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
