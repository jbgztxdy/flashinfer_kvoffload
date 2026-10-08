# Purpose: Verify native or fused MLA Host traffic with warmed inputs.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/profile_glm52_host_25_30.py --mode direct --fraction .25
import argparse
import os

import torch

from bench_glm52_host_comparison import make_wrapper
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "overlap"), default="direct")
    parser.add_argument("--fraction", type=float, default=0.25)
    args = parser.parse_args()
    os.sched_setaffinity(0, set(range(32)) | set(range(64, 96)))
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.set_device(1)
    allocation = MixedHBMHost(1048576, 1)
    cache = allocation.tensor(1)
    cache[:524288].normal_()
    allocation.host_tensor().normal_()
    indices = make_indices(1048576, 2048, [args.fraction], 42, "random")[
        args.fraction
    ].to("cuda:1")
    query = torch.randn((1, 64, 512), dtype=torch.bfloat16, device="cuda:1")
    rope = torch.randn((1, 64, 64), dtype=torch.bfloat16, device="cuda:1")
    output = torch.empty_like(query)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device="cuda:1")
    wrapper = make_wrapper(indices, workspace, 2048)
    kwargs = {"host_page_start": 524288} if args.mode == "overlap" else {}

    def run():
        wrapper.run(
            query, rope, cache[..., :512], cache[..., 512:], out=output, **kwargs
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    print(
        f"mode={args.mode} fraction={args.fraction} Host bytes={round(2048 * args.fraction) * 1152}",
        flush=True,
    )
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
