"""Restore only missing isolated trees from frozen headers, refusing different contents."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'shared_manifest.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
headers={'base':m['base_sha256'],'control_qp':m['control_sha256'],'qks':m['candidate_sha256'],'qks_shared':s['candidate_sha256']}
for v,h in headers.items():
 host=p/(v+'.cuh');math=p/(v+'_mla.cuh');assert sha(host)==h and sha(math)==m['math_sha256']
 d=p/('include_'+v)
 if not d.exists():
  shutil.copytree(root/'include',d);shutil.copy2(host,d/'flashinfer/attention/mla_host_batch.cuh');shutil.copy2(math,d/'flashinfer/attention/mla.cuh')
 assert sha(d/'flashinfer/attention/mla_host_batch.cuh')==h and sha(d/'flashinfer/attention/mla.cuh')==m['math_sha256']
print('ALL_FROZEN_PAIRS_VERIFIED')
