# Purpose: Inspect warm graph phases of direct-to-shared Host attention.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/profile_glm52_host_pcie20.py --fraction .3 --output temp/host_attention_pcie20/phase_30.json
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path

import torch

from bench_glm52_host_comparison import make_wrapper
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices
from flashinfer.jit import gen_batch_mla_module
from flashinfer.mla import _core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    @functools.cache
    def instrumented_module(backend, *module_args, host_prefetch=False):
        specification = gen_batch_mla_module(
            backend, *module_args, host_prefetch=host_prefetch
        )
        if host_prefetch:
            specification.name += f"_private_shared_{args.chunks}_1_phase"
            specification.extra_cuda_cflags.extend(
                [
                    "-DFLASHINFER_MLA_HOST_PRIVATE_SHARED",
                    f"-DFLASHINFER_MLA_HOST_PRIVATE_CHUNKS={args.chunks}",
                    "-DFLASHINFER_MLA_HOST_PRIVATE_PIPELINE",
                    "-DFLASHINFER_MLA_HOST_TRACE",
                ]
            )
        return specification.build_and_load()

    _core.get_batch_mla_module = instrumented_module
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

    def run():
        wrapper.run(
            query,
            rope,
            cache[..., :512],
            cache[..., 512:],
            out=output,
            host_page_start=524288,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(100):
            run()
        end.record()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    phases = wrapper._host_prefetch_state[3608:4040].cpu().view(108, 4).tolist()
    origin = min(row[0] for row in phases)
    result = dict(
        fraction=args.fraction,
        host_chunks=args.chunks,
        repeats=100,
        graph_average_us=start.elapsed_time(end) * 10,
        source_sha256=hashlib.sha256(
            Path("include/flashinfer/attention/mla.cuh").read_bytes()
        ).hexdigest(),
        scope="Instrumented warm graph; timestamps are quantized and tracing changes generated code. Use the paired uninstrumented benchmark for performance claims.",
        phases_ns=phases,
        normalized_phases_us=[
            [(value - origin) / 1000 if value else None for value in row]
            for row in phases
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("phases_ns", "normalized_phases_us")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
