# Purpose: Profile one mixed-residency MLA attention call and its Host traffic.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/profile_glm52_host_overlap.py
import math
import torch
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices
from flashinfer.mla import BatchMLAPagedAttentionWrapper

torch.cuda.set_device(1)
allocation = MixedHBMHost(1048576, 1)
cache = allocation.tensor(1)
cache[:524288].normal_()
allocation.host_tensor().normal_()
indices = make_indices(1048576, 2048, [0.2], 42, 'random')[0.2].to('cuda:1')
q = torch.randn((1, 64, 512), dtype=torch.bfloat16, device='cuda:1')
qpe = torch.randn((1, 64, 64), dtype=torch.bfloat16, device='cuda:1')
wrapper = BatchMLAPagedAttentionWrapper(torch.empty(32*2**20, dtype=torch.uint8, device='cuda:1'), backend='fa2')
wrapper.plan(torch.tensor([0,1], dtype=torch.int32), torch.tensor([0,2048], dtype=torch.int32), indices, torch.tensor([2048], dtype=torch.int32), 64, 512, 64, 1, False, 1/math.sqrt(256), q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
for _ in range(5):
    wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], host_page_start=524288)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], host_page_start=524288)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
