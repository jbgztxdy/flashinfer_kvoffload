"""Keep the independent-process negative controls and shared-allocation diagnosis."""
import hashlib
import json
import statistics
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=json.loads((p/'performance_audit.json').read_text())
rows=[r for r in a['rows'] if r['cohort'].startswith('cached_controls_')]
groups={}
for r in rows:
    key=(r['kind'],r['mtp'],r['requests'],r['setting'],r['fraction'])
    phase=r['cohort'].removeprefix('cached_controls_').removesuffix('_')
    groups.setdefault(key,{})[phase]=r
results=[]
for key,phases in groups.items():
    assert set(phases)=={'before','candidate','after'},key
    before,new,after=[phases[phase] for phase in ['before','candidate','after']]
    mean_base=(before['ours_us']+after['ours_us'])/2
    results.append(dict(kind=key[0],mtp=key[1],requests=key[2],setting=key[3],fraction=key[4],
        base_before_us=before['ours_us'],candidate_us=new['ours_us'],base_after_us=after['ours_us'],
        reduction_vs_base_mean=1-new['ours_us']/mean_base,
        paths={k:v['raw'] for k,v in phases.items()}))
assert len(results)==6
d=json.loads((p/'same_process_control.json').read_text())
assert d['complete'] and d['shared_allocations'] and d['shared_query_indices_cache_and_workspace']
assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
assert d['source_sha256']==sha(p/'same_process_control.py')
snap=p/'gpu3/source_snapshots'
for file in [p/'same_process_control.py',p/'bench_gpu3_native.py']:
    data=file.read_bytes();target=snap/(sha(file)+file.suffix)
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data)
assert d['benchmark_source_sha256']==sha(p/'bench_gpu3_native.py')
means={}
for variant in ['base','cached_ids']:
    records=[r for r in d['records'] if r['variant']==variant]
    assert len(records)==5
    for r in records:
        assert len(r['per_step_us'])==len(r['unique_misses'])==1024
        assert abs(statistics.mean(r['per_step_us'])-r['mean_us'])<1e-8
        assert r['max_abs_error']<.01 and r['cache_payload_checked_rows']==4096
    means[variant]=statistics.mean(r['mean_us'] for r in records)
codegen=json.loads((p/'compiler_cached_normal_summary.json').read_text())
assert codegen['unchanged_original_kernels']==18
result=dict(complete=True,independent_process_controls=results,
    shared_allocation_diagnostic=dict(means_us=means,reduction=1-means['cached_ids']/means['base'],
        raw='same_process_control.json',raw_sha256=sha(p/'same_process_control.json'),
        numerical_and_final_cache_payload_checks=True),
    interpretation='Independent small MTP1 PCIe negative persists about3.9%; same-allocation alternating binaries are effectively equal. Consistent with an allocation/environment effect; causation not established. Original negative records retained. No candidate adopted.',
    goal_complete=False,candidate_adopted=False)
(p/'controls_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
