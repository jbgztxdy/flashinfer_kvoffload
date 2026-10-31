"""Fresh normal BF16/FP16 compilation of the installed production include tree."""
import hashlib
from pathlib import Path
import torch
from flashinfer.jit import env
from flashinfer.jit.attention.modules import gen_batch_mla_module
p=Path(__file__).resolve().parent;root=p.parents[2]
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
env.FLASHINFER_INCLUDE_DIR=root/'include';env.FLASHINFER_CSRC_DIR=root/'csrc';env.FLASHINFER_DATA=root
env.CUTLASS_INCLUDE_DIRS=[root/'3rdparty/cutlass/include',root/'3rdparty/cutlass/tools/util/include']
env.SPDLOG_INCLUDE_DIR=root/'3rdparty/spdlog/include'
for dtype in [torch.bfloat16,torch.float16]:
    for host in [False,True]:
        gen_batch_mla_module('fa2',dtype,dtype,dtype,torch.int32,512,64,False,host_prefetch=host).build_and_load()
print('COMPILED_PRODUCTION_NORMAL')
