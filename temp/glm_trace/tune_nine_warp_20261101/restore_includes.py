"""Restore only missing input trees from frozen host/math pairs."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];m=json.loads((p/'manifest.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for v,h in [('base',m['base_sha256']),('nine_warp',m['candidate_sha256'])]:
 host=p/(v+'.cuh');math=p/(v+'_mla.cuh')
 assert sha(host)==h and sha(math)==m['math_sha256']
 tree=p/('include_'+v)
 if not tree.exists():
  shutil.copytree(root/'include',tree);shutil.copy2(host,tree/'flashinfer/attention/mla_host_batch.cuh');shutil.copy2(math,tree/'flashinfer/attention/mla.cuh')
 assert sha(tree/'flashinfer/attention/mla_host_batch.cuh')==h and sha(tree/'flashinfer/attention/mla.cuh')==m['math_sha256']
print('FROZEN_INPUTS_VERIFIED')
