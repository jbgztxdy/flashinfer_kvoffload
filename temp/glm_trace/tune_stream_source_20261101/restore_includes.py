"""Reconstruct missing private include trees while refusing to overwrite changed ones."""
import hashlib
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
for v in ['base', 'stream_source']:
    tree = p / ('include_' + v)
    if not tree.exists():
        shutil.copytree(root / 'include', tree)
        (tree / 'flashinfer/attention/mla_host_batch.cuh').write_bytes((p / (v + '.cuh')).read_bytes())
        (tree / 'flashinfer/attention/mla.cuh').write_bytes((p / (v + '_mla.cuh')).read_bytes())
    assert sha(tree / 'flashinfer/attention/mla_host_batch.cuh') == sha(p / (v + '.cuh'))
    assert sha(tree / 'flashinfer/attention/mla.cuh') == sha(p / (v + '_mla.cuh'))
