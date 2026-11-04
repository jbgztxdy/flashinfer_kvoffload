"""Compile one isolated candidate without allocating a CUDA device."""
import os
from pathlib import Path
import torch
from flashinfer.jit import env
from flashinfer.jit.attention.modules import gen_batch_mla_module
p=Path(__file__).resolve().parent
variant=os.environ['TUNE_VARIANT']
env.FLASHINFER_CSRC_DIR=p.parents[2]/'csrc'
env.FLASHINFER_DATA=p.parents[2]
env.CUTLASS_INCLUDE_DIRS=[p.parents[2]/'3rdparty/cutlass/include',p.parents[2]/'3rdparty/cutlass/tools/util/include']
env.SPDLOG_INCLUDE_DIR=p.parents[2]/'3rdparty/spdlog/include'
env.FLASHINFER_INCLUDE_DIR=p/('include_'+variant)
dtypes=(torch.bfloat16,)
for dtype in dtypes:
    for host in (True,):
        gen_batch_mla_module('fa2',dtype,dtype,dtype,torch.int32,512,64,False,host_prefetch=host).build_and_load()
print('COMPILED',variant,flush=True)
