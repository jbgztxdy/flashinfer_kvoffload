"""Keep exact tested bytes under the user's focused iteration scope, without broad reruns."""
import hashlib
import json
from pathlib import Path
import shutil
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
m = read(p / 'manifest.json')
base = m['base_sha256']
candidate = m['candidate_sha256']
assert base == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert candidate == '48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73'
assert sha(p / 'base.cuh') == base and sha(p / 'stream_source.cuh') == candidate
assert sha(root / 'include/flashinfer/attention/mla.cuh') == m['math_sha256']
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
scope = read(p / 'scope_change.json')
assert not scope['matrix_started'] and not scope['gpu_process_signalled']
assert read(p / 'extension_state.json')['phase'] == 'stopped_before_matrix_per_user_scope'
state = read(p / 'confirmation_state.json')
assert state['complete'] and state['phase'] == 'local_confirmation_complete'
a = read(p / 'performance_audit.json')
assert not a['incomplete'] and not a['pending_references']
assert len(a['rows']) == 22 and len(a['pairs']) == 11
for tag, settings in [('streamsource_reverse1024_f24_', ['g100_l0', 'g100_l20000', 'g50_l20000']),
                      ('streamsource_forward1024_f24_', ['g100_l0', 'g100_l20000', 'g50_l20000']),
                      ('streamsource_native1024_f24_', ['native_hbm', 'pcie'])]:
    pairs = [r for r in a['pairs'] if r['cohort'] == tag]
    assert {r['setting'] for r in pairs} == set(settings)
    assert all(r['passes'] == 5 and r['positions'] == 1024 for r in pairs)
    assert all(r['reduction'] > 0 for r in pairs if r['setting'].startswith('g100'))
    assert all(r['reduction'] >= -.01 for r in pairs)
assert sum(x['tests'] for x in a['tests'] + a['stress_tests']) == 115
for suffix, n in [('', 38), ('_normal', 76)]:
    proof = read(p / ('compiler_stream_source' + suffix + '_summary.json'))
    assert proof['complete'] and proof['unchanged_original_kernels'] == n
    assert proof['all_original_instruction_and_control_words_identical']
    assert proof['candidate_header_sha256'] == candidate
    for lib in proof['libraries']['stream_source']:
        files = list((p / ('jit_stream_source' + suffix)).rglob(lib['name']))
        assert len(files) == 1 and sha(files[0]) == lib['sha256']
for f in a['files']:
    assert sha(root / f['path']) == f['sha256']
names = ['performance_audit.json', 'confirmation_state.json', 'scope_change.json',
         'compiler_stream_source_summary.json', 'compiler_stream_source_normal_summary.json',
         'tests_stream_source_gpu3.json', 'stress_stream_source_gpu3.json', 'dispatch_probe_stream_source.json',
         'source_review.json', 'stream_source.cuh', 'base.cuh']
snapshot = p / 'focused_retention_snapshot'
snapshot.mkdir(exist_ok=True)
frozen = []
for name in names:
    f = p / name
    target = snapshot / (sha(f) + '_' + name)
    if target.exists():
        assert f.read_bytes() == target.read_bytes()
    else:
        shutil.copy2(f, target)
    frozen.append(dict(path=str(f.relative_to(root)), sha256=sha(f), snapshot=str(target.relative_to(root))))
production = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(production) in [base, candidate], 'Unrelated production edit'
production.write_bytes((p / 'stream_source.cuh').read_bytes())
assert sha(production) == candidate
record = dict(timestamp=time.strftime('%FT%T%z'), candidate_adopted=True, goal_complete=False,
    base_header_sha256=base, candidate_header_sha256=candidate, source_exact_measured_bytes=True,
    validation_scope='Existing115 tests, old38/76fullcodeidentity, actualsingleGraph and first-match3sanitizers, key100GB/50GBfulltwoorders plus nativeHBM/PCIe.',
    explicit_user_preference=scope['explicit_user_instruction'],
    skipped='No156matrix, noSXM/NVLink/H800, no repeated freshroot compile/test. Measuredprivate header is byte-identical to installedroot.',
    risk='Small same-card gains; physicalSXM and other shapes not newly performance-tested. No allscenario guarantee.',
    rows=22, pairs=11, confirmations=[r for r in a['pairs'] if r['positions'] == 1024], frozen=frozen)
(p / 'focused_retention.json').write_text(json.dumps(record, indent=2) + '\n')
print('FOCUSED_RETENTION_COMPLETE', candidate)
