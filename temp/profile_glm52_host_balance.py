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
    parser.add_argument("--local-chunks", type=int, default=16)
    parser.add_argument("--host-chunks", type=int, default=32)
    parser.add_argument("--stream-merge", action="store_true")
    parser.add_argument("--merge-batch", type=int, default=8)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--progressive-merge", action="store_true")
    parser.add_argument("--ready-first", action="store_true")
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
            if args.mixed:
                specification.name += "_mixed_workers48_trace"
                specification.extra_cuda_cflags.append(
                    "-DFLASHINFER_MLA_HOST_MIXED_WORKERS"
                )
            else:
                specification.name += f"_private_shared_{args.host_chunks}_1_local{args.local_chunks}_trace"
                specification.extra_cuda_cflags.extend(
                    [
                        "-DFLASHINFER_MLA_HOST_PRIVATE_SHARED",
                        f"-DFLASHINFER_MLA_HOST_PRIVATE_CHUNKS={args.host_chunks}",
                        f"-DFLASHINFER_MLA_HOST_LOCAL_CHUNKS={args.local_chunks}",
                        "-DFLASHINFER_MLA_HOST_PRIVATE_PIPELINE",
                    ]
                )
            specification.extra_cuda_cflags.append("-DFLASHINFER_MLA_HOST_TRACE")
            if args.stream_merge:
                specification.name += f"_stream_merge_{args.merge_batch}"
                specification.extra_cuda_cflags.extend(
                    [
                        "-DFLASHINFER_MLA_HOST_STREAM_MERGE",
                        f"-DFLASHINFER_MLA_HOST_MERGE_BATCH={args.merge_batch}",
                    ]
                )
            if args.progressive_merge:
                specification.name += f"_progressive_merge_{args.merge_batch}"
                specification.extra_cuda_cflags.extend(
                    [
                        "-DFLASHINFER_MLA_HOST_PROGRESSIVE_MERGE",
                        f"-DFLASHINFER_MLA_HOST_MERGE_BATCH={args.merge_batch}",
                    ]
                )
                if args.ready_first:
                    specification.name += "_ready_first"
                    specification.extra_cuda_cflags.append(
                        "-DFLASHINFER_MLA_HOST_READY_FIRST"
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
        "local_chunks": args.local_chunks,
        "host_chunks": args.host_chunks,
        "stream_merge": args.stream_merge,
        "merge_batch": args.merge_batch,
        "mixed": args.mixed,
        "progressive_merge": args.progressive_merge,
        "ready_first": args.ready_first,
        "residency": allocation.residency,
        "source_sha256": hashlib.sha256(
            Path("include/flashinfer/attention/mla.cuh").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    compute_ctas = 48 if args.mixed else args.local_chunks + args.host_chunks
    merge_capacity = 108 - compute_ctas
    heads_per_cta = 1 if merge_capacity >= 64 else (2 if merge_capacity >= 32 else 4)
    merge_ctas = 64 // heads_per_cta
    for fraction in (0.25, 0.3):
        indices.copy_(selected[fraction])
        for _ in range(10):
            graph.replay()
        samples = []
        for _ in range(args.rounds):
            graph.replay()
            trace = wrapper._host_prefetch_state[3608:4040].cpu().reshape(108, 4)
            origin = trace[:, 0].min().item()
            if args.mixed:
                local_end = (trace[:compute_ctas, 2].max().item() - origin) / 1000
                host_end = (trace[:compute_ctas, 3].max().item() - origin) / 1000
                host_qk_ready = None
                host_prepare_end = (
                    trace[:compute_ctas, 1].max().item() - origin
                ) / 1000
            else:
                local_end = (trace[: args.local_chunks, 3].max().item() - origin) / 1000
                host_end = (
                    trace[args.local_chunks : compute_ctas, 3].max().item() - origin
                ) / 1000
                host_qk_ready = (
                    trace[args.local_chunks : compute_ctas, 2].max().item() - origin
                ) / 1000
                host_prepare_end = (
                    trace[args.local_chunks : compute_ctas, 1].max().item() - origin
                ) / 1000
            merge_end = (
                trace[compute_ctas : compute_ctas + merge_ctas, 3].max().item() - origin
            ) / 1000
            host_trace = (
                trace[:compute_ctas, 3]
                if args.mixed
                else trace[args.local_chunks : compute_ctas, 3]
            )
            samples.append(
                {
                    "local_end_us": local_end,
                    "host_prepare_end_us": host_prepare_end,
                    "host_end_us": host_end,
                    "merge_end_us": merge_end,
                    "post_compute_us": merge_end - max(local_end, host_end),
                    "host_cta_spread_us": (
                        host_trace.max().item() - host_trace.min().item()
                    )
                    / 1000,
                }
            )
            if host_qk_ready is not None:
                samples[-1]["host_qk_ready_us"] = host_qk_ready
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
