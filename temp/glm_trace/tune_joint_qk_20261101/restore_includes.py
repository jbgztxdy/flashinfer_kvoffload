"""Restore missing private input trees, never overwrite a different frozen pair."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
m=json.loads((p/'manifest.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for v,h in [('base',m['base_sha256']),('joint_mdo_typed',m['joint_mdo_control_sha256']),('joint_qk',m['candidate_sha256'])]:
    assert sha(p/(v+'.cuh'))==h and sha(p/(v+'_mla.cuh'))==m['math_sha256']
    for suffix in ['', '_normal']:
        tree=p/('include_'+v+suffix)
        if not tree.exists():
            shutil.copytree(root/'include',tree)
            shutil.copy2(p/(v+'.cuh'),tree/'flashinfer/attention/mla_host_batch.cuh')
            shutil.copy2(p/(v+'_mla.cuh'),tree/'flashinfer/attention/mla.cuh')
        assert sha(tree/'flashinfer/attention/mla_host_batch.cuh')==h
        assert sha(tree/'flashinfer/attention/mla.cuh')==m['math_sha256']
print('ALL_PRIVATE_HOST_MATH_PAIRS_VERIFIED')
