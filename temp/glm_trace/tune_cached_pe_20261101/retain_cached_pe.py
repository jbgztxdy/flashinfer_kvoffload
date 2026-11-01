"""Retain the exact measured header only after complete frozen regression and confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
parser = argparse.ArgumentParser()
parser.add_argument('--install', action='store_true')
args = parser.parse_args()
base = '7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
candidate = 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
math = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(p / 'production_before_cached_pe.cuh') == base
assert sha(p / 'cached_pe.cuh') == candidate
assert sha(root / 'include/flashinfer/attention/mla.cuh') == sha(p / 'cached_pe_mla.cuh') == math
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
producer = read(p / 'broader_validation_state.json')
if producer['phase'] == 'failed':
    assert producer['error'] == 'Failed matrix_sxm_native.log'
    original = read(p / 'resource_preflight_failure/broader_validation_state.json')
    assert producer == original
    assert all(x['exit_code'] == 0 for x in producer['steps'][:-1])
    assert producer['steps'][-1]['exit_code'] == 1
    failure = read(p / 'resource_preflight_failure.json')
    for item in failure['preserved']:
        assert sha(p / 'resource_preflight_failure' / item['name']) == item['sha256']
    producer = read(p / 'matrix_resume_state.json')
    assert producer['preserved_failure_sha256'] == sha(p / 'resource_preflight_failure.json')
assert producer['complete'] and producer['phase'] == 'regression_measurements_complete'
assert all(x['exit_code'] == 0 for x in producer['steps'])
a = read(p / 'performance_audit.json')
c = read(p / 'coverage.json')
assert c['audit_sha256'] == sha(p / 'performance_audit.json')
assert c['gpu3_matrix_complete'] and c['sxm_matrix_complete']
assert len(c['groups']) == 4 and sum(x['expected_cells'] for x in c['groups']) == 156
assert not a['incomplete'] and not a['pending_references']
assert sum(x['tests'] for x in a['tests'] + a['stress_tests']) == 111
for f in a['files']:
    assert sha(root / f['path']) == f['sha256']
confirmations = []
for tag in ['cachedreverse1024_f24_', 'cachedforward1024_f24_', 'cachednative1024_f24_',
            'cachedsxmreverse1024_f24_', 'cachedsxmforward1024_f24_', 'cachedsxmnative1024_f24_']:
    pairs = [r for r in a['pairs'] if r['cohort'] == tag]
    assert len(pairs) == (2 if 'native' in tag else 3)
    assert all(r['positions'] == 1024 and r['passes'] == 5 and r['fetch_ctas'] == 24 and r['mtp'] == 8 and r['requests'] == 128 for r in pairs)
    assert all(r['reduction'] > -.01 for r in pairs)
    assert all(r['reduction'] > 0 for r in pairs if r['setting'].startswith('g100'))
    confirmations.extend(pairs)
controls = read(p / 'controls_audit.json')
assert controls['complete'] and controls['all_original_negatives_preserved'] and controls['no_over_one_percent_negative_reproduced']
def key(r):
    return r['cohort'], r['kind'], r['mtp'], r['requests'], r['setting'], r['fraction']
allowed = {key(r['original_point']) for r in controls['controls']}
assert allowed == {('cached_pe_shapes_gpu3_native_', 'lru', 1, 1, 'native_hbm', None),
                   ('cached_pe_shapes_gpu3_native_', 'lru', 1, 256, 'native_hbm', None)}
slow = [r for r in a['pairs'] if r['cohort'].startswith('cached_pe_shapes_') and r['reduction'] < -.01]
assert {key(r) for r in slow} <= allowed, slow
for item in controls['controls']:
    assert item['original_point'] in a['pairs']
    before, new, after = item['raws']
    assert all(new['mean_us'] <= x['mean_us'] * 1.01 for x in [before, after])
    for r in item['raws']:
        assert sha(root / r['raw']) == r['sha256']
changed = [r for r in a['pairs'] if r['cohort'].startswith('cached_pe_shapes_') and r['kind'] == 'lru' and r['mtp'] * r['requests'] >= 432]
assert len(changed) == 60 and all(r['reduction'] > -.01 for r in changed)
assert statistics.mean(r['reduction'] for r in changed) > 0
for suffix, n in [('', 36), ('_normal', 72)]:
    d = read(p / ('compiler_cached_pe' + suffix + '_summary.json'))
    assert d['complete'] and d['unchanged_original_kernels'] == n and d['candidate_header_sha256'] == candidate
    for v, libs in d['libraries'].items():
        directory = p.parent / 'tune_joint_qk_20261101' / ('jit_joint_qk' + suffix) if v == 'base' else p / ('jit_cached_pe' + suffix)
        for lib in libs:
            matches = list(directory.rglob(lib['name']))
            assert len(matches) == 1 and sha(matches[0]) == lib['sha256']
words = read(p / 'machine_word_identity.json')
assert words['complete'] and [x['original_kernel_instances'] for x in words['builds']] == [36, 72]
assert all(x['all_words_identical'] for x in words['builds'])
jq = p.parent / 'tune_joint_qk_20261101'
bridge = read(jq / 'machine_word_identity.json')
assert bridge['complete'] and all(x['cleanup_all_words_identical'] for x in bridge['builds'])
for name, count in [('tests_cached_pe_gpu3.json', 105), ('stress_cached_pe_gpu3.json', 6)]:
    d = read(p / name)
    assert d['complete'] and d['exit_code'] == 0 and d['header_sha256'] == candidate
    assert d['resources_start']['foreign'] == d['resources_end']['foreign'] == []
    log = p / ('pytest_cached_pe_gpu3.log' if count == 105 else 'pytest_stress_cached_pe.log')
    assert sha(log) == d['log_sha256'] and f'{count} passed' in log.read_text()
for f in (p / 'dispatch_probes').glob('*.json'):
    d = read(f)
    assert d['header_sha256'] == candidate and d['kernel_nodes'] == 1
    launch = d['launches'][0]
    assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 4, 2]
    assert launch['dynamic_shared_bytes'] == 147968 and launch['active_blocks_per_SM'] == 1
for kind in ['synccheck', 'racecheck']:
    assert 'SUMMARY: 0' in (p / (kind + '_cached_pe.log')).read_text()
production = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(production) in [base, candidate], 'Unrelated production edit'
names = ['performance_audit.json', 'coverage.json', 'controls_audit.json', 'controls_plan.json', 'manifest.json',
         'sxm_confirmation_plan.json', 'machine_word_identity.json', 'compiler_cached_pe_summary.json',
         'compiler_cached_pe_normal_summary.json', 'tests_cached_pe_gpu3.json', 'stress_cached_pe_gpu3.json',
         'dispatch_probe_cached_pe.json', 'static_instruction_review.json', 'broader_validation_state.json',
         'cached_pe.cuh', 'cached_pe_mla.cuh', 'production_before_cached_pe.cuh']
if (p / 'matrix_resume_state.json').exists():
    names += ['matrix_resume_state.json', 'resource_preflight_failure.json']
frozen = []
snapshot = p / 'retention_snapshot'
snapshot.mkdir(exist_ok=True)
for name in names:
    f = p / name
    target = snapshot / (sha(f) + '_' + name)
    if target.exists():
        assert target.read_bytes() == f.read_bytes()
    else:
        shutil.copy2(f, target)
    frozen.append(dict(path=str(f.relative_to(root)), sha256=sha(f), snapshot=str(target.relative_to(root))))
record = dict(timestamp=time.strftime('%FT%T%z'), candidate_adopted=bool(args.install), goal_complete=False,
    base_header_sha256=base, measured_header_sha256=candidate, candidate_header_sha256=candidate,
    source_policy='Install exact measured a436 header without cleanup or dispatch parameter changes.',
    matrix_cells=156, matrix_pairs=78, changed_lru_pairs=60, confirmations=confirmations,
    changed_matrix_reduction_range=[min(x['reduction'] for x in changed), max(x['reduction'] for x in changed)],
    changed_matrix_mean_reduction=statistics.mean(x['reduction'] for x in changed),
    slow_original_cells=slow, control_brackets=controls['controls'], frozen=frozen,
    limitation='All original negatives preserved. Brackets do not establish unique cause or promise each allocation faster. H800 unvalidated; all root validation required after install; overall goal active.')
if args.install:
    production.write_bytes((p / 'cached_pe.cuh').read_bytes())
    assert sha(production) == candidate
(p / 'cached_pe_retention.json').write_text(json.dumps(record, indent=2) + '\n')
print('CACHED_PE_RETENTION', record['candidate_adopted'], record['changed_matrix_mean_reduction'])
