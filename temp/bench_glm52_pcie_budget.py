# Purpose: Measure sparse pinned-Host reads for the 25% and 30% PCIe budget.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_pcie_budget.py --output temp/host_attention_pcie20/sparse_fetch.json
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import torch

from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices
from bench_glm52_sparse_decode_p2p import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", type=int, default=[131072, 1048576])
    parser.add_argument("--blocks", nargs="+", type=int, default=[60, 64, 108, 216])
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    os.environ["TMPDIR"] = str(root / "temp")
    os.sched_setaffinity(0, set(range(32)) | set(range(64, 96)))
    torch.set_num_threads(8)
    torch.cuda.set_device(1)
    torch.manual_seed(42)
    cubin = root / "temp/bench_sparse_host_fetch.cubin"
    subprocess.run(
        [
            "/usr/local/cuda-13.1/bin/nvcc",
            "-O3",
            "-arch=sm_80",
            "-cubin",
            str(root / "temp/bench_sparse_host_fetch.cu"),
            "-o",
            str(cubin),
        ],
        check=True,
    )
    driver = ctypes.CDLL("libcuda.so.1")

    def check(status):
        if status:
            raise RuntimeError(f"CUDA Driver error {status}")

    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    check(driver.cuModuleLoad(ctypes.byref(module), str(cubin).encode()))
    check(driver.cuModuleGetFunction(ctypes.byref(function), module, b"sparse_fetch"))
    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        source_sha256=hashlib.sha256(
            (root / "temp/bench_sparse_host_fetch.cu").read_bytes()
        ).hexdigest(),
        timing="Median GPU cycle inside a graph of 100 read-and-store calls; excludes index updates and initialization.",
        results=[],
    )
    for context in args.contexts:
        allocation = MixedHBMHost(context, 1)
        cache = allocation.tensor(1)
        host = allocation.host_tensor()
        host.normal_()
        indices = torch.empty(2048, dtype=torch.int32, device="cuda:1")
        output = torch.empty((2048, 576), dtype=torch.bfloat16, device="cuda:1")
        values = [
            ctypes.c_uint64(cache.data_ptr() + context // 2 * 1152),
            ctypes.c_uint64(indices.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_int(context // 2),
        ]
        params = (ctypes.c_void_p * 4)(*[ctypes.addressof(value) for value in values])
        fractions = [0, 0.25, 0.3]
        selections = [
            make_indices(context, 2048, fractions, 42 + seed, "random")
            for seed in range(8)
        ]
        for blocks in args.blocks:
            stream = torch.cuda.Stream(device=1)

            def launch():
                check(
                    driver.cuLaunchKernel(
                        function,
                        blocks,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        ctypes.c_void_p(stream.cuda_stream),
                        params,
                        None,
                    )
                )

            indices.copy_(selections[0][0.25])
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                launch()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph, stream=stream):
                    start.record()
                    for _ in range(100):
                        launch()
                    end.record()
                samples = {fraction: [] for fraction in fractions}
                rng = random.Random(context + blocks)
                for iteration in range(args.rounds):
                    order = fractions.copy()
                    rng.shuffle(order)
                    for fraction in order:
                        selected = selections[iteration % 8][fraction]
                        indices.copy_(selected)
                        stream.synchronize()
                        graph.replay()
                        end.synchronize()
                        graph.replay()
                        end.synchronize()
                        samples[fraction].append(start.elapsed_time(end) * 10)
                        if iteration == 0 and fraction:
                            mask = selected >= context // 2
                            torch.testing.assert_close(
                                output[mask.to("cuda:1")].cpu(),
                                host[selected[mask].long() - context // 2, 0],
                                rtol=0,
                                atol=0,
                            )
                for fraction in fractions:
                    summary = summarize(samples[fraction])
                    result["results"].append(
                        dict(
                            context_tokens=context,
                            blocks=blocks,
                            fraction=fraction,
                            host_bytes=round(2048 * fraction) * 1152,
                            latency=summary,
                            verified=True,
                        )
                    )
                    print(
                        context,
                        blocks,
                        fraction,
                        round(summary["median_us"], 3),
                        flush=True,
                    )
            del graph, start, end, cache, host
            gc.collect()
            cache = allocation.tensor(1)
            host = allocation.host_tensor()
        del cache, host, indices, output
        gc.collect()
        torch.cuda.synchronize()
        allocation.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
