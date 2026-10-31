"""Compile both tested dtypes in the normal build without allocating a GPU."""
import os
from pathlib import Path
import torch
from flashinfer.jit import env
from flashinfer.jit.attention.modules import gen_batch_mla_module
p=Path(__file__).resolve().parent
root=p.parents[2]
env.FLASHINFER_DATA=root
env.FLASHINFER_CSRC_DIR=root/'csrc'
env.FLASHINFER_INCLUDE_DIR=p/'include_final'
env.CUTLASS_INCLUDE_DIRS=[root/'3rdparty/cutlass/include',root/'3rdparty/cutlass/tools/util/include']
env.SPDLOG_INCLUDE_DIR=root/'3rdparty/spdlog/include'
for dtype in (torch.bfloat16,torch.float16):
 for host in (False,True):
  gen_batch_mla_module('fa2',dtype,dtype,dtype,torch.int32,512,64,False,host_prefetch=host).build_and_load()
print('NORMAL_BUILDS_COMPILED',flush=True)
