"""Verify completed evidence without treating rejected extension plans as coverage."""
import hashlib
import json
import time
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
expected = {
    'base': '7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469',
    'claim_probe': '95ff4054a653135668daa8dc41b6fe6abbcb285b82da1254eb66c795be4f5c1c',
    'claim_store': '15aa954ac63ce52024439446ca9dde583875eaf5afa24fd8df343075ecd9c81e',
}
math = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root / 'include/flashinfer/attention/mla_host_batch.cuh') == expected['base']
assert sha(root / 'include/flashinfer/attention/mla.cuh') == math
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
for v, h in expected.items():
    assert sha(p / (v + '.cuh')) == h
    assert sha(p / (v + '_mla.cuh')) == math
    tree = p / ('include_' + v) / 'flashinfer/attention'
    if tree.exists():
        assert sha(tree / 'mla_host_batch.cuh') == h and sha(tree / 'mla.cuh') == math
reused = read(p / 'reused_production_base.json')
assert reused['complete'] and reused['additional_tests_counted'] == 0
for item in reused['proofs']:
    assert sha(root / item['path']) == item['sha256']
proofs = []
for v in ['claim_probe', 'claim_store']:
    for suffix, count in [('', 36), ('_normal', 72)]:
        f = p / ('compiler_' + v + suffix + '_summary.json')
        d = read(f)
        assert d['complete'] and d['unchanged_original_kernels'] == count
        assert d['all_original_instruction_and_control_words_identical']
        assert d['base_header_sha256'] == expected['base'] and d['candidate_header_sha256'] == expected[v]
        assert d['math_header_sha256'] == math and d['all_new_stack_and_local_zero']
        for variant, libs in d['libraries'].items():
            directory = (p.parent / 'tune_joint_qk_20261101' /
                         ('jit_production_joint_normal' if suffix else 'jit_retained_joint_qk')) if variant == 'base' else p / ('jit_' + variant + suffix)
            for lib in libs:
                matches = list(directory.rglob(lib['name']))
                assert len(matches) == 1 and sha(matches[0]) == lib['sha256']
        proofs.append(dict(path=f.name, sha256=sha(f), count=count))
    d = read(p / ('dispatch_probe_' + v + '.json'))
    launch = d['launches'][0]
    assert d['kernel_nodes'] == 1 and d['header_sha256'] == expected[v]
    assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 4, 2]
    assert launch['dynamic_shared_bytes'] == 147968 and launch['active_blocks_per_SM'] == 1
    assert d['resources_start']['foreign'] == d['resources_end']['foreign'] == []
    for kind in ['synccheck', 'racecheck', 'memcheck']:
        log = (p / (kind + '_' + v + '.log')).read_text()
        assert 'ERROR SUMMARY: 0 errors' in log or (kind == 'racecheck' and 'RACECHECK SUMMARY: 0 hazards' in log)
for name in ['validation_state.json', 'validation_store_state.json']:
    d = read(p / name)
    assert d['complete'] and all(x['exit_code'] == 0 for x in d['steps'])
a = read(p / 'performance_audit.json')
assert len(a['rows']) == 28 and len(a['pairs']) == 14
assert not a['incomplete'] and not a['pending_references']
assert a['timed_launches'] == 22784 and a['accepted_segments'] == 1424
assert sum(x['tests'] for x in a['tests'] + a['stress_tests']) == 226
for item in a['files']:
    assert sha(root / item['path']) == item['sha256']
short = read(p / 'short_performance_audit.json')
for row in short['rows']:
    assert [r for r in a['rows'] if r['raw'] == row['raw']] == [row]
for item in short['files']:
    assert sha(root / item['path']) == item['sha256']
extension = read(p / 'extension_state.json')
assert extension['phase'] == 'failed' and not extension['complete']
assert all(x['exit_code'] == 0 for x in extension['steps'])
forward = [r for r in a['pairs'] if r['cohort'] == 'claim_store_forward1024_f24_' and r['setting'].startswith('g100')]
assert len(forward) == 2 and all(r['reduction'] < 0 for r in forward)
assert not (p / 'coverage.json').exists(), 'No unexecuted matrix may be claimed as completed'
review = read(p / 'atomic_source_review.json')
assert len(set(x['normalized_lifecycle_kernel_sha256'] for x in review['source'])) == 1
failure_files = sorted(f for f in (p / 'test_fixture_failure').iterdir() if f.is_file())
assert len(failure_files) >= 6 and any(f.suffix == '.log' for f in failure_files)
record = dict(checkpoint_verified=True, timestamp=time.strftime('%FT%T%z'),
    core_validation_complete=True, experiment_complete=True, regression_measurements_complete=False,
    candidates_adopted=False, goal_complete=False, production_unchanged=True, headers=expected, proofs=proofs,
    rows=28, pairs=14, timed_launches=22784, accepted_segments=1424, unique_tests=226,
    extension_result='Rejected by positive100GB forward confirmation gate; preserve terminal1, do not restart.',
    missing_for_adoption=['No broad matrix or SXM validation for either rejected candidate.'],
    failures=[dict(path=str(f.relative_to(p)), sha256=sha(f)) for f in failure_files],
    limitation='GPU3 HBM/PCIe only. Sanitizers first matched BF16/K97 launch. Static atomic counts are not dynamic performance causality.')
(p / 'checkpoint_audit.json').write_text(json.dumps(record, indent=2) + '\n')
print('CHECKPOINT_VERIFIED', record['rows'], record['unique_tests'])
