"""Restore frozen private includes, verifying existing trees without overwriting them."""
import argparse
import hashlib
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
a = argparse.ArgumentParser()
a.add_argument('--variants', nargs='+', choices=['base', 'warp_rows'],
               default=['base', 'warp_rows'])
a = a.parse_args()
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
canonical_math = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
math = p/'gpu3/source_snapshots'/(canonical_math+'.cuh')
assert sha(math) == canonical_math
for variant in a.variants:
    host = p/(variant+'.cuh')
    dest = p/('include_'+variant)
    if dest.exists():
        assert (dest/'flashinfer/attention/mla_host_batch.cuh').read_bytes() == host.read_bytes()
        assert (dest/'flashinfer/attention/mla.cuh').read_bytes() == math.read_bytes()
        print('VERIFIED_EXISTING', variant)
        continue
    shutil.copytree(root/'include', dest)
    (dest/'flashinfer/attention/mla_host_batch.cuh').write_bytes(host.read_bytes())
    (dest/'flashinfer/attention/mla.cuh').write_bytes(math.read_bytes())
    print('RESTORED', variant, sha(host), sha(math))
