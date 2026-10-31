"""Rebuild ignored private include trees from frozen sources; never overwrite a tree."""
import argparse
import hashlib
from pathlib import Path
import shutil

here=Path(__file__).resolve().parent;root=here.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--round',choices=['wait','heads','qk','qk_loop','cached_runtime'],default='cached_runtime')
parser.add_argument('--variants',nargs='+')
a=parser.parse_args()
p=here.parent/('tune_'+a.round+'_20261031')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
canonical_math='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
variants=a.variants or [f.stem for f in sorted(p.glob('*.cuh')) if not f.stem.endswith('_mla')]
for variant in variants:
    host=p/(variant+'.cuh');assert host.is_file()
    math=p/(variant+'_mla.cuh')
    if not math.exists():
        options=[p/'base_mla.cuh',p/'source_snapshots'/(canonical_math+'.cuh'),
                 p/'gpu3/source_snapshots'/(canonical_math+'.cuh'),root/'include/flashinfer/attention/mla.cuh']
        matches=[f for f in options if f.exists() and sha(f)==canonical_math]
        assert matches,'Missing frozen canonical math header'
        math=matches[0]
    dest=p/('include_'+variant)
    if dest.exists():
        assert (dest/'flashinfer/attention/mla_host_batch.cuh').read_bytes()==host.read_bytes(),dest
        assert (dest/'flashinfer/attention/mla.cuh').read_bytes()==math.read_bytes(),dest
        print('VERIFIED_EXISTING',variant)
        continue
    shutil.copytree(root/'include',dest)
    (dest/'flashinfer/attention/mla_host_batch.cuh').write_bytes(host.read_bytes())
    (dest/'flashinfer/attention/mla.cuh').write_bytes(math.read_bytes())
    print('RESTORED',variant,sha(host),sha(math))
