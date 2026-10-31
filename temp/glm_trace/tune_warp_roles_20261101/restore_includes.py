"""Restore only missing private include trees; reject any differing existing input."""
import hashlib
import json
from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
m=json.loads((p/'manifest.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for v,host,math in [('base',m['base_sha256'],m['canonical_math_sha256']),('warp_roles',m['candidate_sha256'],m['candidate_math_sha256']),('warp_roles_isolated',json.loads((p/'isolated_manifest.json').read_text())['candidate_sha256'],json.loads((p/'isolated_manifest.json').read_text())['candidate_math_sha256'])]:
    assert sha(p/(v+'.cuh'))==host and sha(p/(v+'_mla.cuh'))==math
    for suffix in ['', '_normal']:
        tree=p/('include_'+v+suffix)
        if not tree.exists():
            shutil.copytree(root/'include',tree)
            shutil.copy2(p/(v+'.cuh'),tree/'flashinfer/attention/mla_host_batch.cuh')
            shutil.copy2(p/(v+'_mla.cuh'),tree/'flashinfer/attention/mla.cuh')
        assert sha(tree/'flashinfer/attention/mla_host_batch.cuh')==host
        assert sha(tree/'flashinfer/attention/mla.cuh')==math
print('ALL_PRIVATE_HOST_MATH_PAIRS_VERIFIED')
