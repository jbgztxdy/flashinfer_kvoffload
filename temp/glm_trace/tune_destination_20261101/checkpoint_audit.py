"""Verify rejected measurements while withholding the incompatible hit diagnostic."""
import hashlib
import json
from pathlib import Path
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
m = read(p / 'manifest.json')
assert sha(p / 'base.cuh') == m['base_sha256'] == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(root / 'include/flashinfer/attention/mla_host_batch.cuh') == m['base_sha256']
assert sha(p / 'final_destination.cuh') == m['candidate_sha256'] == '3c744e78251fdc6787c6a68c0d248773857c859e1c09dba687c58ef1dfaa7749'
assert sha(root / 'include/flashinfer/attention/mla.cuh') == m['math_sha256'] == sha(p / 'final_destination_mla.cuh')
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
reused = read(p / 'reused_cached_pe_control.json')
assert reused['complete'] and reused['additional_tests_counted'] == 0
for item in reused['proofs']:
    assert sha(root / item['path']) == item['sha256']
proofs = []
for suffix, count in [('', 38), ('_normal', 76)]:
    f = p / ('compiler_final_destination' + suffix + '_summary.json')
    d = read(f)
    assert d['complete'] and d['unchanged_original_kernels'] == count
    assert d['all_original_instruction_and_control_words_identical'] and d['all_new_stack_and_local_zero']
    assert d['candidate_header_sha256'] == m['candidate_sha256']
    assert all('REG:255 STACK:0 SHARED:2048 LOCAL:0' in r['resource'] for r in d['new_kernels'])
    for v, libs in d['libraries'].items():
        folder = p.parent / 'tune_cached_pe_20261101' / ('jit_cached_pe' + suffix) if v == 'base' else p / ('jit_' + v + suffix)
        for lib in libs:
            matches = list(folder.rglob(lib['name']))
            assert len(matches) == 1 and sha(matches[0]) == lib['sha256']
    proofs.append(dict(file=f.name, sha256=sha(f), original_instances=count))
probe = read(p / 'dispatch_probe_final_destination.json')
assert probe['kernel_nodes'] == 1 and probe['header_sha256'] == m['candidate_sha256']
launch = probe['launches'][0]
assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 4, 2]
assert launch['dynamic_shared_bytes'] == 147968 and launch['active_blocks_per_SM'] == 1
for kind in ['synccheck', 'racecheck', 'memcheck']:
    log = (p / (kind + '_final_destination.log')).read_text()
    assert 'ERROR SUMMARY: 0 errors' in log or (kind == 'racecheck' and 'RACECHECK SUMMARY: 0 hazards' in log)
s = read(p / 'validation_state.json')
assert s['phase'] == 'failed' and s['error'] == 'Failed audit_performance.log'
assert all(x['exit_code'] == 0 for x in s['steps'][:-1]) and s['steps'][-1]['exit_code'] == 1
a = read(p / 'performance_audit.json')
assert len(a['rows']) == 6 and len(a['pairs']) == 3
assert a['timed_launches'] == 1152 and a['accepted_segments'] == 72
assert sum(x['tests'] for x in a['tests'] + a['stress_tests']) == 113
assert not a['incomplete'] and not a['pending_references']
assert all(x['reduction'] < 0 for x in a['pairs'])
assert len(a['invalid_reference_hit_diagnostics']) == 3
assert all(r['reference_hit_rate'] is None for r in a['rows'] if r['variant'] == 'final_destination')
for item in a['files']:
    assert sha(root / item['path']) == item['sha256']
for v in ['base', 'final_destination']:
    d = read(p / ('remap_diagnosis_' + v + '.json'))
    assert d['complete'] and not d['performance_claim'] and d['header_sha256'] == sha(p / (v + '.cuh'))
    assert d['source_sha256'] == sha(p / 'diagnose_remap.py')
    assert d['resources_start']['foreign'] == d['resources_end']['foreign'] == []
    assert [x['before_launch_reference_misses'] for x in d['steps']] == [41904, 0, 41904, 0]
    assert [x['kernel_unique_counter'] for x in d['steps']] == [97, 0, 97, 0]
    assert [x['legacy_postlaunch_threshold_count'] for x in d['steps']] == ([0] * 4 if v == 'final_destination' else [41904, 0, 41904, 0])
model = read(p / 'destination_model_review.json')
assert model['complete'] and model['cpu_only'] and model['case_count'] == 144
assert model['source_sha256'] == sha(p / 'review_destination_model.py')
review = read(p / 'source_review.json')
assert len(review['math_helpers_exactly_preserved']) == 6
result = dict(checkpoint_verified=True, timestamp=time.strftime('%FT%T%z'), candidate_adopted=False,
    goal_complete=False, experiment_complete=True, broad_regression_complete=False, rows=6, pairs=3,
    timed_launches=1152, accepted_segments=72, independent_tests=113, proofs=proofs,
    invalid_reference_hit_rows=3, extra_diagnostic_launches=8, cpu_structural_cases=144,
    outcome='Reject all three negative pairs; no full/native/SXM expansion.',
    limitation='Reference hit rate withheld for candidate, cannot recover historical trace hit rate. Unique counter/timing/two references checked independently; four-step diagnoses are not fulltrace proof.')
(p / 'checkpoint_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('CHECKPOINT_VERIFIED', result['rows'], result['independent_tests'])
