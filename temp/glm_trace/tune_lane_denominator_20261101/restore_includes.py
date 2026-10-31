"""Rebuild missing private include trees from frozen headers, refusing different inputs."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
m = json.loads((p / 'manifest.json').read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
for variant, h in [('base', m['base_sha256']),
                   ('joint_qk', m['joint_qk_control_sha256']),
                   ('lane_d', m['candidate_sha256'])]:
    host = p / (variant + '.cuh')
    math = p / (variant + '_mla.cuh')
    assert sha(host) == h and sha(math) == m['math_sha256']
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        shutil.copy2(host, d / 'flashinfer/attention/mla_host_batch.cuh')
        shutil.copy2(math, d / 'flashinfer/attention/mla.cuh')
    assert sha(d / 'flashinfer/attention/mla_host_batch.cuh') == h
    assert sha(d / 'flashinfer/attention/mla.cuh') == m['math_sha256']
print('ALL_FROZEN_PAIRS_VERIFIED')
