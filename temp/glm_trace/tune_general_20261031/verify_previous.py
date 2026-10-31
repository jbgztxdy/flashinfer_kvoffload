"""Re-run the previous round's read-only sample audit without regenerating its artifacts."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
old=p.parent/'tune_100gb_20261030'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
report=old/'report.py'
namespace={'__file__':str(report)}
exec(compile(report.read_text().split('neighbor_rows=')[0],str(report),'exec'),namespace)
a=json.loads((old/'audit.json').read_text());s=json.loads((old/'summary.json').read_text())
assert len(namespace['rows'])==s['all_audited_cells']==577
assert namespace['launches']==s['timed_launches']==944064
assert namespace['guard_count']==s['accepted_segments']==59760
assert namespace['max_error']==s['sampled_max_abs_error']
assert namespace['rows']==s['rows']
for f in a['files']:assert sha(root/f['path'])==f['sha256'],f
for f,h in a['source_sha256'].items():assert sha(old/f)==h,f
for f,h in ((old/'summary.json',a['summary_sha256']),(old/'results.csv',a['csv_sha256']),
            (root/'docs/mla_host_a100_100gb_optimization_20261030.md',a['document_sha256']),
            (root/'tests/attention/test_mla_host_batch.py',a['test_source_sha256']),
            (old/'pytest_final.log',a['test_log_sha256']),
            (root/'docs/assets/glm53_100gb_optimization_20261030/trace_optimization.png',a['figure_sha256']),
            (old/'figure_data.json',a['figure_data_sha256']),
            (root/'include/flashinfer/attention/mla_host_batch.cuh',a['production_header_sha256'])):
    if f==root/'tests/attention/test_mla_host_batch.py' and sha(f)!=h:
        f=p/'source_snapshots'/(h+'.py')
    elif f==root/'include/flashinfer/attention/mla_host_batch.cuh' and sha(f)!=h:
        f=old/'final.cuh'
    assert sha(f)==h,str(f)
result=dict(checked_at=time.strftime('%FT%T%z'),read_only_audit=True,
    previous_round_verified=True,cells=len(namespace['rows']),timed_launches=namespace['launches'],
    accepted_segments=namespace['guard_count'],max_abs_error=namespace['max_error'],
    raw_files=len(a['files']),header_sha256=a['production_header_sha256'],
    current_production_header_sha256=sha(root/'include/flashinfer/attention/mla_host_batch.cuh'),
    current_test_source_sha256=sha(root/'tests/attention/test_mla_host_batch.py'),
    original_sources_verified_from_frozen_snapshots=True,previous_audit_sha256=sha(old/'audit.json'),test_log=(old/'pytest_final.log').read_text())
(p/'previous_reaudit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
