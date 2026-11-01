"""Bind the current progress decision to audited rows and actual resumed producers."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
a = read(p / 'performance_audit.json')
c = read(p / 'coverage.json')
state = read(p / ('matrix_resume_state.json' if (p / 'matrix_resume_state.json').exists() else 'broader_validation_state.json'))
complete = state['complete'] and c['gpu3_matrix_complete'] and c['sxm_matrix_complete']
final = read(p / 'final_retention_audit.json') if (p / 'final_retention_audit.json').exists() else {}
adopted = final.get('complete', False) and final.get('candidate_adopted', False)
if adopted:
    assert sha(p.parents[2] / 'include/flashinfer/attention/mla_host_batch.cuh') == final['production_header_sha256']
live = []
if not state['complete'] and state['phase'] != 'failed':
    ps = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'pid,ppid,etimes,args'], text=True, capture_output=True)
    assert ps.returncode == 0
    live = ps.stdout.splitlines()[1:]
record = dict(recorded_at=time.strftime('%FT%T%z'), status='measurements_complete_pending_retention' if complete else 'pending_native_matrix_resource',
    conclusion='完整确认和受限矩阵已完成；最后正常链路矩阵按原计划补缺，尚未安装。' if not complete else '完整156矩阵及两顺序完整确认完成；等待保留门槛及生产root新验证。',
    candidate_adopted=False, goal_complete=False, audit_sha256=sha(p / 'performance_audit.json'),
    measured_cells=len(a['rows']), pairs=len(a['pairs']), matrix_cells_audited=sum(g['completed_audited_cells'] for g in c['groups']),
    missing_matrix_cells=sum(len(g['missing_cells']) for g in c['groups']),
    original_negative_samples_preserved=True, controls_audit_sha256=sha(p / 'controls_audit.json'),
    original_parent_terminal_exit_code=1 if (p / 'resource_preflight_failure.json').exists() else None,
    live_measurement_producers=live,
    remaining=['Complete missing native cells with idle GPU0/1', 'Audit every negative; frozen retention gate', 'Fresh production root compile/111 tests/codegen76/Graph after install'])
(p / 'decisions.json').write_text(json.dumps(record, indent=2) + '\n')
if adopted:
    record.update(status='retained_and_fresh_root_verified', candidate_adopted=True,
                  conclusion='完整156矩阵和所有门槛通过，已保留实测a436原件；生产root111测试/76实例代码/单kernel Graph验证完成。',
                  final_retention_audit_sha256=sha(p / 'final_retention_audit.json'), remaining=['Overall goal active: continue reducing the large100GB gap.'])
    (p / 'decisions.json').write_text(json.dumps(record, indent=2) + '\n')
print(record['status'], record['matrix_cells_audited'], record['missing_matrix_cells'])
