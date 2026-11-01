"""Verify current production against frozen measurements and independent root validation."""
import hashlib
import json
from pathlib import Path
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
statefile = p / ('completion_resume_state.json' if (p / 'completion_resume_state.json').exists() else 'completion_state.json')
s = read(statefile)
assert s['complete'] and s['phase'] == 'complete' and s['candidate_adopted']
assert all(x['exit_code'] == 0 for x in s['steps'])
r = read(p / 'cached_pe_retention.json')
assert r['candidate_adopted'] and not r['goal_complete']
assert sha(root / 'include/flashinfer/attention/mla_host_batch.cuh') == sha(p / 'cached_pe.cuh') == r['candidate_header_sha256']
assert r['candidate_header_sha256'] == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(root / 'include/flashinfer/attention/mla.cuh') == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
for item in r['frozen']:
    assert sha(root / item['snapshot']) == item['sha256']
frozen = {Path(x['path']).name: root / x['snapshot'] for x in r['frozen']}
a = read(frozen['performance_audit.json'])
c = read(frozen['coverage.json'])
assert c['audit_sha256'] == sha(frozen['performance_audit.json'])
assert c['gpu3_matrix_complete'] and c['sxm_matrix_complete']
assert sum(x['expected_cells'] for x in c['groups']) == 156
assert len(a['rows']) == 200 and len(a['pairs']) == 97
assert not a['incomplete'] and not a['pending_references']
for item in a['files']:
    assert sha(root / item['path']) == item['sha256']
t = read(p / 'tests_production_gpu3.json')
assert t['complete'] and t['exit_code'] == 0 and t['expected_tests'] == 111
assert t['production_header_sha256'] == r['candidate_header_sha256']
assert Path(t['include_path']) == root / 'include'
assert Path(t['fresh_workspace']) == p / 'jit_production_cached_pe_normal'
assert sha(root / 'tests/attention/test_mla_host_batch.py') == t['test_source_sha256']
assert sha(p / 'test_cofetch_progress.py') == t['stress_source_sha256']
assert t['resources_start']['foreign'] == t['resources_end']['foreign'] == []
assert t['log_sha256'] == sha(p / 'pytest_production_gpu3.log')
assert '111 passed' in (p / 'pytest_production_gpu3.log').read_text()
code = read(p / 'production_codegen_normal_summary.json')
assert code['complete'] and code['kernel_instances'] == 76
assert code['all_original_and_new_instruction_control_words_and_resources_identical']
for lib in code['libraries']:
    paths = list((p / 'jit_production_cached_pe_normal').rglob(lib['name']))
    assert len(paths) == 1 and sha(paths[0]) == lib['sha256']
probe = read(p / 'dispatch_probe_production_cached_pe.json')
assert probe['header_sha256'] == r['candidate_header_sha256'] and probe['kernel_nodes'] == 1
assert probe['resources_start']['foreign'] == probe['resources_end']['foreign'] == []
cfg = probe['launches'][0]
assert cfg['grid'] == [1, 108, 1] and cfg['block'] == [32, 4, 2]
assert cfg['dynamic_shared_bytes'] == 147968 and cfg['active_blocks_per_SM'] == 1
assert 'BatchMLACachedPEKernel' in cfg['kernel_name']
for name, digest in r['production_validation'].items():
    assert sha(p / name) == digest
result = dict(complete=True, timestamp=time.strftime('%FT%T%z'), candidate_adopted=True, goal_complete=False,
    production_header_sha256=r['candidate_header_sha256'], matrix_cells=156, rows=200, pairs=97,
    production_tests=111, production_kernel_instances=76, single_kernel_graph=True,
    retained_source='Exact measured a436 bytes, no unmeasured cleanup.',
    remaining_goal='Large100GB still clearly above device/per-step max reference; H800 not validated; continue optimization.',
    historical_failure_preserved='Original15217/47513 terminal1 at native resourcepreflight; resumed missing20 separately.')
(p / 'final_retention_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('FINAL_RETENTION_VERIFIED', result['rows'], result['production_tests'])
