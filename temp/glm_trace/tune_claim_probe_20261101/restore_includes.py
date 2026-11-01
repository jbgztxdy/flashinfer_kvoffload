"""Rebuild missing frozen includes, refusing to overwrite different existing trees."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'store_manifest.json').read_text())
for variant,h in {'base':m['base_sha256'],'claim_probe':m['candidate_sha256'],'claim_store':s['candidate_sha256']}.items():
 host=p/(variant+'.cuh');math=p/(variant+'_mla.cuh');assert sha(host)==h and sha(math)==m['math_sha256']
 d=p/('include_'+variant)
 if not d.exists():
  shutil.copytree(root/'include',d);shutil.copy2(host,d/'flashinfer/attention/mla_host_batch.cuh');shutil.copy2(math,d/'flashinfer/attention/mla.cuh')
 assert sha(d/'flashinfer/attention/mla_host_batch.cuh')==h and sha(d/'flashinfer/attention/mla.cuh')==m['math_sha256']
print('ALL_FROZEN_PAIRS_VERIFIED')
