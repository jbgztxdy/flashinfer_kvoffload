# Purpose: Record local, Host, and merge completion times for mixed-tier MLA.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/profile_glm52_host_balance.py --output temp/host_attention_balance/phases.json
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from bench_glm52_host_comparison import MixedHBMHost, make_indices, make_wrapper
from flashinfer.jit import gen_batch_mla_module
from flashinfer.mla import _core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    @functools.cache
    def traced_module(backend, *module_args, host_prefetch=False):
        specification = gen_batch_mla_module(
            backend, *module_args, host_prefetch=host_prefetch
        )
        if host_prefetch:
            specification.name += "_private_shared_32_1_local16_trace"
            specification.extra_cuda_cflags.extend(
                [
                    "-DFLASHINFER_MLA_HOST_PRIVATE_SHARED",
                    "-DFLASHINFER_MLA_HOST_PRIVATE_CHUNKS=32",
                    "-DFLASHINFER_MLA_HOST_LOCAL_CHUNKS=16",
                    "-DFLASHINFER_MLA_HOST_PRIVATE_PIPELINE",
                    "-DFLASHINFER_MLA_HOST_TRACE",
                ]
            )
        return specification.build_and_load()

    _core.get_batch_mla_module = traced_module
    os.sched_setaffinity(0, set(range(32, 64)) | set(range(96, 128)))
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.manual_seed(91)
    device = f"cuda:{args.device}"
    allocation = MixedHBMHost(1048576, args.device)
    cache = allocation.tensor(args.device)
    host = allocation.host_tensor()
    cache[:524288].normal_()
    host.normal_()
    query = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    rope = torch.randn((1, 64, 64), device=device, dtype=torch.bfloat16)
    indices = torch.empty(2048, device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    workspace = torch.empty(32 * 2**20, device=device, dtype=torch.uint8)
    wrapper = make_wrapper(indices, workspace, 2048)
    selected = make_indices(1048576, 2048, [0.25, 0.3], 42, "random")
    indices.copy_(selected[0.25])

    def run(wrapper, cache):
        wrapper.run(
            query,
            rope,
            cache[..., :512],
            cache[..., 512:],
            out=output,
            host_page_start=524288,
        )

    run(wrapper, cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(wrapper, cache)
    result = {
        "scope": "Instrumented single-call globaltimer traces; excluded from latency comparisons.",
        "device": args.device,
        "residency": allocation.residency,
        "source_sha256": hashlib.sha256(
            Path("include/flashinfer/attention/mla.cuh").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    for fraction in (0.25, 0.3):
        indices.copy_(selected[fraction])
        for _ in range(10):
            graph.replay()
        samples = []
        for _ in range(args.rounds):
            graph.replay()
            trace = wrapper._host_prefetch_state[3608:4040].cpu().reshape(108, 4)
            origin = trace[:, 0].min().item()
            local_end = (trace[:16, 3].max().item() - origin) / 1000
            host_end = (trace[16:48, 3].max().item() - origin) / 1000
            merge_end = (trace[48:80, 3].max().item() - origin) / 1000
            samples.append(
                {
                    "local_end_us": local_end,
                    "host_end_us": host_end,
                    "merge_end_us": merge_end,
                    "post_compute_us": merge_end - max(local_end, host_end),
                }
            )
        case = {
            "host_fraction": fraction,
            "median_us": {
                key: statistics.median(sample[key] for sample in samples)
                for key in samples[0]
            },
            "samples": samples,
        }
        result["cases"].append(case)
        print(json.dumps(case["median_us"]), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    del graph, wrapper, cache, host
    allocation.close()


if __name__ == "__main__":
    main()
