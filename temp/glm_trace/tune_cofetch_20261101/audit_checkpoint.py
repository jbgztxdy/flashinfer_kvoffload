"""Verify the completed integrated-fetch/control evidence without changing production."""
import hashlib
import json
from pathlib import Path
import subprocess

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
read = lambda f: json.loads(f.read_text())
production = 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
math = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
matrix = 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh') == production
assert sha(root/'include/flashinfer/attention/mla.cuh') == math
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json') == matrix
assert not read(p/'continuation_state.json')['active_runners']
assert not read(p/'decisions.json')['candidate_adopted']
audit = read(p/'performance_audit.json')
assert len(audit['rows']) == 9 and len(audit['pairs']) == 6
assert audit['timed_launches'] == 1728 and audit['accepted_segments'] == 108
assert not audit['pending_references'] and not audit['incomplete']
assert sum(t['tests'] for t in audit['tests']) == 210
assert sum(t['tests'] for t in audit['stress_tests']) == 4
assert read(p/'decisions.json')['audit_sha256'] == sha(p/'performance_audit.json')
for variant, dynamic in [('cofetch', 111104), ('stage1_control', 147968)]:
    header = sha(p/(variant+'.cuh'))
    for suffix, originals in [('',34),('_normal',68)]:
        code = read(p/('compiler_'+variant+suffix+'_summary.json'))
        assert code['complete'] and code['unchanged_original_kernels'] == originals
        assert code['base_header_sha256'] == production
        assert code['candidate_header_sha256'] == header and code['math_header_sha256'] == math
        assert code['all_new_stack_and_local_zero']
    probe = read(p/('dispatch_probe_'+variant+'.json'))
    assert probe['header_sha256'] == header and probe['math_header_sha256'] == math
    assert probe['graph_nodes'] == probe['kernel_nodes'] == 1
    config = probe['launches'][0]
    assert config['grid'] == [1,108,1] and config['block'] == [32,4,2]
    assert config['dynamic_shared_bytes'] == dynamic and config['active_blocks_per_SM'] == 1
names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
    '--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
snapshots = 0
for name in set(names):
    f = root/name
    if f.suffix == '.json': read(f)
    if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
        assert sha(f) == f.stem
        snapshots += 1
for f in p.glob('*.py'): compile(f.read_text(),str(f),'exec')
result = dict(complete=True,goal_status='active',candidate_adopted=False,normal_tests=214,
    performance_rows=9,paired_comparisons=6,timed_launches=1728,accepted_segments=108,
    verified_source_snapshots=snapshots,production_header_sha256=production,
    math_header_sha256=math,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
