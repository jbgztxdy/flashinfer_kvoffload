"""Use the frozen candidate after the repository-path pytest plugin has configured JIT."""
import os
from pathlib import Path
from flashinfer.jit import env
p=Path(__file__).resolve().parent
variant=os.environ['TUNE_VARIANT']
assert variant in ('base','lane_d','joint_qk')
env.FLASHINFER_INCLUDE_DIR=p/('include_'+variant)
assert (env.FLASHINFER_INCLUDE_DIR/'flashinfer/attention/mla_host_batch.cuh').read_bytes()==(p/(variant+'.cuh')).read_bytes()
