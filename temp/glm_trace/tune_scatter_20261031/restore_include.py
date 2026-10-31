"""Recreate an absent isolated include tree without overwriting a pending build's inputs."""
import argparse,hashlib,re,shutil
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
a=argparse.ArgumentParser();a.add_argument('variant');v=a.parse_args().variant
assert re.fullmatch('[a-z0-9_]+',v)
header=p/(v+'.cuh');assert header.is_file()
include=p/('include_'+v)
target=include/'flashinfer/attention/mla_host_batch.cuh'
if include.exists():
    assert target.read_bytes()==header.read_bytes(),'Existing isolated include differs: do not overwrite queued source'
else:
    shutil.copytree(root/'include',include)
    target.write_bytes(header.read_bytes())
print(v,hashlib.sha256(target.read_bytes()).hexdigest())
