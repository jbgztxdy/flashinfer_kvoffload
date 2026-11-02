"""Recreate absent private includes from frozen source; never overwrite a different tree."""
from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
for variant in ['base','qk_cursor']:
 tree=p/('include_'+variant)
 if not tree.exists():
  shutil.copytree(root/'include',tree)
  (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes((p/(variant+'.cuh')).read_bytes())
  (tree/'flashinfer/attention/mla.cuh').write_bytes((p/(variant+'_mla.cuh')).read_bytes())
 assert (tree/'flashinfer/attention/mla_host_batch.cuh').read_bytes()==(p/(variant+'.cuh')).read_bytes()
 assert (tree/'flashinfer/attention/mla.cuh').read_bytes()==(p/(variant+'_mla.cuh')).read_bytes()
print('FROZEN_INCLUDES_VERIFIED')
