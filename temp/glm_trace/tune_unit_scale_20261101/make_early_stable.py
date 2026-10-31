"""Branch on unchanged finite maxima before computing the unused rescale exponent."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
host = (p/'stable_scale.cuh').read_text()
math = (p/'stable_scale_mla.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(host) == 'de3659aefaef3664fafad2e56d0bb52d2eaf7c08689615d3b2ddc23391f914cf'
assert sha(math) == 'ace386627cf39b1fcc1ff7112a642ada2a7f56a1d4d9fec504432478dda30ae8'
candidate = host.replace('StableScaleHostTraits','EarlyStableScaleHostTraits').replace(
    'StableScaleHostBatchParams','EarlyStableScaleHostBatchParams')
old = '''      float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
      // The computed scale, including NaN/non-unit cases, is unchanged.
      // Equal finite maxima need no mathematical rescale. The old fused expression
      // can round slightly away from 1; FP32 rounding may therefore differ.
      if (!MLAUnitOScaleGuard<KTraits>::value || (m_prev[j] != m[j] || !isfinite(o_scale))) {'''
assert math.count(old) == 2
private = math.replace(old, '''      // For an unchanged finite maximum, mathematical rescaling is exactly unit.
      // Check that state before the SFU operation; score exponentials stay native.
      if (!MLAUnitOScaleGuard<KTraits>::value || m_prev[j] != m[j] || !isfinite(m[j])) {
        float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);''')
variant = 'early_stable'
assert not (p/(variant+'.cuh')).exists()
(p/(variant+'.cuh')).write_text(candidate)
(p/(variant+'_mla.cuh')).write_text(private)
shutil.copytree(root/'include',p/('include_'+variant))
(p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(candidate)
(p/('include_'+variant)/'flashinfer/attention/mla.cuh').write_text(private)
for name in ['test_candidate.py','candidatepaths.py','screen.py','stress_candidate.py']:
    f=p/name;text=f.read_text()
    if name in ['screen.py','stress_candidate.py']:
        text=text.replace("['unit_scale', 'stable_scale']", "['unit_scale', 'stable_scale', 'early_stable']")
        text=text.replace("['base', 'unit_scale', 'stable_scale']", "['base', 'unit_scale', 'stable_scale', 'early_stable']")
    else:
        text=text.replace("'stable_scale'", "'stable_scale','early_stable'")
    f.write_text(text)
for stem in ['compiler','probe']:
    text=(p/(stem+'_stable_scale.py')).read_text().replace('stable_scale','early_stable')
    text=text.replace('StableScaleHostTraits','EarlyStableScaleHostTraits').replace(
        'StableScaleHostBatchParams','EarlyStableScaleHostBatchParams')
    (p/(stem+'_'+variant+'.py')).write_text(text)
(p/'early_manifest.json').write_text(json.dumps(dict(
    predecessor_host_sha256=sha(host),predecessor_math_sha256=sha(math),
    candidate_sha256=sha(candidate),candidate_math_sha256=sha(private),
    mechanism='Same native QK/P/PV/layout/fetch/LRU. On unchanged finite maximum, skip o_scale exponent and d/O scaling. Compute original rescale only on changed/nonfinite maximum. Finite-state attention equation unchanged; FP32 rescale rounding differs.',
    evidence='stable_scale keeps exponent/finite-scale dependency before its guard and remains slower than base. This changes the guard dependency and removes unused SFU work; no geometry/cadence/unroll sweep.',
    candidate_adopted=False,goal_complete=False),indent=2)+'\n')
print((p/'early_manifest.json').read_text())
