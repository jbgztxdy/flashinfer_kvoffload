"""Use unchanged online-softmax maxima rather than exact rounded-exp2 equality."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
host = (p/'unit_scale.cuh').read_text()
math = (p/'unit_scale_mla.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(host) == 'b8bc7b16c7583697d1afa5697e87fd732e082b8503388646059a351c7a314593'
assert sha(math) == 'a95a8f467bc3da4d52fd30125026e786f499bd20ee16356dcd8d4e209bf6e528'
candidate = host.replace('UnitScaleHostTraits','StableScaleHostTraits').replace(
    'UnitScaleHostBatchParams','StableScaleHostBatchParams')
assert math.count('o_scale != 1.0f') == 2
private = math.replace('o_scale != 1.0f', '(m_prev[j] != m[j] || !isfinite(o_scale))')
private = private.replace('// Exact multiplication by 1 leaves finite d/O unchanged.',
    '// Equal finite maxima need no mathematical rescale. The old fused expression\n'
    '      // can round slightly away from 1; FP32 rounding may therefore differ.')
variant = 'stable_scale'
assert not (p/(variant+'.cuh')).exists()
(p/(variant+'.cuh')).write_text(candidate)
(p/(variant+'_mla.cuh')).write_text(private)
shutil.copytree(root/'include',p/('include_'+variant))
(p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(candidate)
(p/('include_'+variant)/'flashinfer/attention/mla.cuh').write_text(private)
for name in ['test_candidate.py','candidatepaths.py','screen.py','stress_candidate.py']:
    f=p/name;text=f.read_text()
    if name == 'screen.py':
        text=text.replace("choices=['unit_scale']", "choices=['unit_scale', 'stable_scale']")
        text=text.replace("['base', 'unit_scale']", "['base', 'unit_scale', 'stable_scale']")
    elif name == 'stress_candidate.py':
        text=text.replace("choices=['unit_scale']", "choices=['unit_scale', 'stable_scale']")
    else:
        text=text.replace("'unit_scale'", "'unit_scale','stable_scale'")
    f.write_text(text)
for stem in ['compiler','probe']:
    text=(p/(stem+'_unit_scale.py')).read_text().replace('unit_scale','stable_scale')
    text=text.replace('UnitScaleHostTraits','StableScaleHostTraits').replace(
        'UnitScaleHostBatchParams','StableScaleHostBatchParams')
    (p/(stem+'_stable_scale.py')).write_text(text)
f=p/'audit_performance.py';text=f.read_text()
old="                assert executed_math == 'a95a8f467bc3da4d52fd30125026e786f499bd20ee16356dcd8d4e209bf6e528'"
assert text.count(old)==1
text=text.replace(old,"                assert executed_math == sha(p/(cfg['variant']+'_mla.cuh'))")
f.write_text(text)
manifest=dict(predecessor_host_sha256=sha(host),predecessor_math_sha256=sha(math),
    candidate_sha256=sha(candidate),candidate_math_sha256=sha(private),
    mechanism='Same layout/fetch/LRU/nativeQK/P/PV/tileorder as unit_scale. Rescale d/O iff max changed or computed scale is nonfinite. When maxima match, mathematical exp2(delta) is1; FP32 rounding differs from the original contracted expression. Original nonfinite-scale behavior preserved.',
    evidence='unit_scale is slower6.6-7.7% at100GB. PrimarySASS has FFMA->EX2->unitcomparison/BRA; isolated same-expression probe returns26440nonunit of65536equalmax pairs, maxerror3.5763e-7. Probe doesnot measure actual attention branch frequency.',
    candidate_adopted=False,goal_complete=False)
(p/'stable_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
