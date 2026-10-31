"""Audit the three frozen, rejected candidates and their completed evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

p = Path(__file__).resolve().parent
root = p.parents[2]
pair = p.parent/'tune_qk_pair_20261031'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
read = lambda f: json.loads(f.read_text())
production_sha = 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
math_sha = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
matrix_sha = 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh') == production_sha
assert sha(root/'include/flashinfer/attention/mla.cuh') == math_sha
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json') == matrix_sha
file_count = json_count = snapshot_count = 0
for directory in [pair, p]:
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others',
        '--exclude-standard', '--', str(directory.relative_to(root))], cwd=root, text=True)
    for name in set(names.splitlines()):
        f = root/name
        file_count += 1
        if f.suffix == '.json':
            read(f)
            json_count += 1
        if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
            assert sha(f) == f.stem, f
            snapshot_count += 1
    for f in directory.glob('*.py'):
        compile(f.read_text(), str(f), 'exec')
    state = read(directory/'continuation_state.json')
    assert state['goal_status'] == 'active' and not state['active_runners']
    assert not read(directory/'decisions.json')['candidate_adopted']
tests = []
for directory, variant, compiler, probe, stack in [
    (pair, 'qk_pair', 'compiler_pair', 'dispatch_probe', 0),
    (pair, 'qk_pair_prefetch', 'compiler_prefetch', 'dispatch_probe_prefetch', 8),
    (p, 'warp_rows', 'compiler_warp_rows', 'dispatch_probe', 0),
]:
    header_sha = sha(directory/(variant+'.cuh'))
    test = read(directory/('tests_'+variant+'_gpu3.json'))
    assert test['complete'] and test['exit_code'] == test['compile_exit_code'] == 0
    assert test['header_sha256'] == header_sha
    assert test['test_source_sha256'] == sha(root/'tests/attention/test_mla_host_batch.py')
    for suffix, original_count in [('', 34), ('_normal', 68)]:
        code = read(directory/(compiler+suffix+'_summary.json'))
        assert code['complete'] and code['unchanged_original_kernels'] == original_count
        assert code['candidate_header_sha256'] == header_sha
        assert code['base_header_sha256'] == production_sha
        assert code['math_header_sha256'] == math_sha
        assert all(f'STACK:{stack} ' in r['resource'] and 'LOCAL:0 ' in r['resource']
                   for r in code['new_kernels'])
    launch = read(directory/(probe+'.json'))
    assert launch['header_sha256'] == header_sha
    assert launch['math_header_sha256'] == math_sha
    assert launch['graph_nodes'] == launch['kernel_nodes'] == 1
    config = launch['launches'][0]
    assert config['grid'] == [1, 108, 1] and config['block'] == [32, 4, 2]
    assert config['dynamic_shared_bytes'] == 147968 and config['active_blocks_per_SM'] == 1
    tests.append(dict(variant=variant, tests=105, header_sha256=header_sha, stack_bytes=stack))
rows = pairs = timed = segments = 0
for directory, expected_rows, expected_pairs in [(pair, 12, 6), (p, 6, 3)]:
    audit = read(directory/'performance_audit.json')
    assert not audit['pending_references'] and not audit['incomplete']
    assert len(audit['rows']) == expected_rows and len(audit['pairs']) == expected_pairs
    assert audit['production_header_sha256'] == production_sha
    assert audit['original_matrix_sha256'] == matrix_sha
    assert read(directory/'decisions.json')['audit_sha256'] == sha(directory/'performance_audit.json')
    rows += expected_rows
    pairs += expected_pairs
    timed += audit['timed_launches']
    segments += audit['accepted_segments']
result = dict(complete=True, goal_status='active', candidate_adopted=False,
    source_files=file_count, json_files=json_count, verified_source_snapshots=snapshot_count,
    tests=tests, normal_tests=315, performance_rows=rows, paired_comparisons=pairs,
    timed_launches=timed, accepted_segments=segments,
    production_header_sha256=production_sha, math_header_sha256=math_sha,
    original_matrix_sha256=matrix_sha)
(p/'checkpoint_audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
