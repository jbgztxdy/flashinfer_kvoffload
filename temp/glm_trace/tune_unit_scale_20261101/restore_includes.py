"""Restore frozen host/private math pairs, verifying existing trees without overwriting."""
import argparse
import hashlib
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
a = argparse.ArgumentParser()
a.add_argument('--variants', nargs='+', choices=['base','unit_scale','stable_scale','early_stable'],
               default=['base','unit_scale','stable_scale','early_stable'])
a = a.parse_args()
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
for variant in a.variants:
    host, math = p/(variant+'.cuh'), p/(variant+'_mla.cuh')
    expected = {
        'base':'34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3',
        'unit_scale':'a95a8f467bc3da4d52fd30125026e786f499bd20ee16356dcd8d4e209bf6e528',
        'stable_scale':'ace386627cf39b1fcc1ff7112a642ada2a7f56a1d4d9fec504432478dda30ae8',
        'early_stable':'a33aacba4cb43fc02c8cf3badc56cdb5cca9e8143ae883b593e7e2d512978565',
    }[variant]
    assert sha(math) == expected
    dest = p/('include_'+variant)
    if dest.exists():
        assert (dest/'flashinfer/attention/mla_host_batch.cuh').read_bytes() == host.read_bytes()
        assert (dest/'flashinfer/attention/mla.cuh').read_bytes() == math.read_bytes()
        print('VERIFIED_EXISTING',variant)
        continue
    shutil.copytree(root/'include',dest)
    (dest/'flashinfer/attention/mla_host_batch.cuh').write_bytes(host.read_bytes())
    (dest/'flashinfer/attention/mla.cuh').write_bytes(math.read_bytes())
    print('RESTORED',variant,sha(host),sha(math))
