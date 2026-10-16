# Purpose: One native and one batched (no Host pages) MLA launch for Nsight Compute comparison.
# Usage: ncu --nvtx ... python temp/profile_host_batch_local.py --rows 1024
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_glm52_decode  # noqa: E402,F401
from flashinfer.mla import BatchMLAPagedAttentionWrapper  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--rows", type=int, default=1024)
parser.add_argument("--fetch", type=int, default=8)
args = parser.parse_args()
device = torch.device("cuda", args.device)
torch.cuda.set_device(device)
torch.manual_seed(0)
pages = 8 * 2**20
cache = torch.randn(pages, 1, 576, dtype=torch.bfloat16, device=device)
rows = args.rows
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=device)
qpe = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=device)
out = torch.empty_like(q)
indices = torch.randint(0, pages, (rows * 2048,), dtype=torch.int32, device=device)
wrapper = BatchMLAPagedAttentionWrapper(torch.empty(512 * 2**20, dtype=torch.uint8, device=device), backend="fa2")
wrapper.plan(torch.arange(rows + 1, dtype=torch.int32), torch.arange(rows + 1, dtype=torch.int32) * 2048, indices,
             torch.full((rows,), 2048, dtype=torch.int32), 64, 512, 64, 1, False, 1 / 16,
             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
wrapper.host_fetch_ctas = args.fetch
for boundary in (None, pages, None, pages):
    wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=boundary)
torch.cuda.synchronize()
