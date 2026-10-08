# Purpose: Compare serial pinned-H2D copy, native Host reads, and fused Host overlap.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_comparison.py --layout random --output temp/host_comparison_random.json
import argparse
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost
from bench_glm52_sparse_decode_p2p import (
    AccessDescriptor,
    AllocationProperties,
    MemoryLocation,
    MixedHBM,
    gpu_snapshot,
    make_indices,
    summarize,
)
from flashinfer.mla import BatchMLAPagedAttentionWrapper


class StagedHBM(MixedHBM):
    """Alias existing local HBM and append a compact HBM DMA destination."""

    def __init__(self, source, topk):
        self.source = source
        self.driver = source.driver
        self.address = ctypes.c_uint64()
        self.handles = []
        self.mappings = []
        self.tokens = source.local_tokens + topk
        self.local_tokens = source.local_tokens
        self.local = source.local
        self.bytes_per_token = source.bytes_per_token
        self.nbytes = self.tokens * self.bytes_per_token
        self.closed = False
        local_bytes = source.local_tokens * self.bytes_per_token
        staged_bytes = topk * self.bytes_per_token
        prop = AllocationProperties()
        prop.type = 1
        prop.location = MemoryLocation(1, self.local)
        granularity = ctypes.c_size_t()
        self._call(
            "cuMemGetAllocationGranularity",
            ctypes.byref(granularity),
            ctypes.byref(prop),
            ctypes.c_int(0),
        )
        if local_bytes % granularity.value:
            raise ValueError("Local HBM alias size must satisfy VMM granularity")
        staged_bytes = (
            (staged_bytes + granularity.value - 1)
            // granularity.value
            * granularity.value
        )
        self.nbytes = local_bytes + staged_bytes
        self._call(
            "cuMemAddressReserve",
            ctypes.byref(self.address),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_size_t(granularity.value),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0),
        )
        try:
            handle = ctypes.c_uint64()
            self._call(
                "cuMemCreate",
                ctypes.byref(handle),
                ctypes.c_size_t(staged_bytes),
                ctypes.byref(prop),
                ctypes.c_uint64(0),
            )
            self.handles.append(handle)
            for offset, size, allocation in (
                (0, local_bytes, source.handles[0]),
                (local_bytes, staged_bytes, handle),
            ):
                address = self.address.value + offset
                self._call(
                    "cuMemMap",
                    ctypes.c_uint64(address),
                    ctypes.c_size_t(size),
                    ctypes.c_size_t(0),
                    allocation,
                    ctypes.c_uint64(0),
                )
                self.mappings.append((address, size))
            access = (AccessDescriptor * 1)(
                AccessDescriptor(MemoryLocation(1, self.local), 3)
            )
            self._call(
                "cuMemSetAccess",
                self.address,
                ctypes.c_size_t(self.nbytes),
                access,
                ctypes.c_size_t(1),
            )
        except Exception:
            self.close()
            raise


def make_wrapper(indices, workspace, topk):
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, topk], dtype=torch.int32),
        indices,
        torch.tensor([topk], dtype=torch.int32),
        64,
        512,
        64,
        1,
        False,
        1 / 16,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return wrapper


def capture_cycle(
    wrapper,
    query,
    rope,
    cache,
    output,
    repeats,
    host_start=None,
    source=None,
    destination=None,
    rows=0,
):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(repeats):
            if rows:
                destination[:rows].copy_(source[:rows], non_blocking=True)
            wrapper.run(
                query,
                rope,
                cache[..., :512],
                cache[..., 512:],
                out=output,
                host_page_start=host_start,
            )
        end.record()
    return graph, start, end


def replay(captured, repeats):
    graph, start, end = captured
    graph.replay()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def benchmark_context(args, tokens):
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    allocation = MixedHBMHost(tokens, args.device)
    cache = allocation.tensor(args.device)
    local = cache[: tokens // 2]
    host = allocation.host_tensor()
    local.normal_()
    host.normal_()
    alias = StagedHBM(allocation, args.topk)
    alias_cache = alias.tensor(args.device)
    staged = alias_cache[tokens // 2 :]
    packed = torch.zeros((args.topk, 1, 576), dtype=torch.bfloat16, pin_memory=True)
    assert packed.is_pinned() and not packed.is_cuda
    query = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((1, 64, 64), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    direct_indices = torch.empty(args.topk, dtype=torch.int32, device=device)
    copy_indices = torch.empty_like(direct_indices)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
    copy_workspace = torch.empty_like(workspace)
    direct_wrapper = make_wrapper(direct_indices, workspace, args.topk)
    copy_wrapper = make_wrapper(copy_indices, copy_workspace, args.topk)
    index_sets = [
        make_indices(tokens, args.topk, args.fractions, args.seed + i, args.layout)
        for i in range(args.index_sets)
    ]
    prepared = []
    for selected_set in index_sets:
        case_set = {}
        for fraction, selected in selected_set.items():
            mask = selected >= tokens // 2
            host_ids = selected[mask].long() - tokens // 2
            remapped = selected.clone()
            remapped[mask] = tokens // 2 + torch.arange(
                len(host_ids), dtype=torch.int32
            )
            case_set[fraction] = (selected, host_ids, remapped)
        prepared.append(case_set)
    direct_indices.copy_(index_sets[0][0.0])
    copy_indices.copy_(index_sets[0][0.0])
    for _ in range(5):
        direct_wrapper.run(query, rope, cache[..., :512], cache[..., 512:], out=output)
        direct_wrapper.run(
            query,
            rope,
            cache[..., :512],
            cache[..., 512:],
            out=output,
            host_page_start=tokens // 2,
        )
        copy_wrapper.run(
            query, rope, alias_cache[..., :512], alias_cache[..., 512:], out=output
        )
    torch.cuda.synchronize(device)
    modes = ["copy", "direct", "overlap"]
    current_module = direct_wrapper._host_prefetch_module
    previous_module = None
    if args.baseline_module:
        import tvm_ffi

        previous_module = tvm_ffi.load_module(str(args.baseline_module.resolve()))
        modes.append("previous")
    graphs = {}
    for mode in modes:
        direct_wrapper._host_prefetch_module = (
            previous_module if mode == "previous" else current_module
        )
        graphs[mode] = {}
        fractions = args.fractions if mode == "copy" else [0.0]
        for fraction in fractions:
            rows = round(fraction * args.topk) if mode == "copy" else 0
            wrapper = copy_wrapper if mode == "copy" else direct_wrapper
            data = alias_cache if mode == "copy" else cache
            kwargs = dict(
                host_start=tokens // 2 if mode in ("overlap", "previous") else None,
                source=packed,
                destination=staged,
                rows=rows,
            )
            graphs[mode][fraction] = {
                "hot": capture_cycle(
                    wrapper, query, rope, data, output, args.repeats, **kwargs
                ),
                "single": capture_cycle(
                    wrapper, query, rope, data, output, 1, **kwargs
                ),
            }
    direct_wrapper._host_prefetch_module = current_module
    eviction = torch.empty(64 * 2**20, dtype=torch.uint8, device=device)
    samples = {
        fraction: {
            mode: {"gpu_cycle": [], "cold_single": [], "serial_wall": []}
            for mode in modes
        }
        for fraction in args.fractions
    }
    packing = {fraction: [] for fraction in args.fractions}
    errors = {fraction: {} for fraction in args.fractions}
    before = gpu_snapshot()
    rng = random.Random(args.seed + tokens)
    for iteration in range(args.rounds):
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            selected, host_ids, remapped = prepared[iteration % args.index_sets][
                fraction
            ]
            rows = len(host_ids)
            direct_indices.copy_(selected)
            copy_indices.copy_(remapped)
            torch.cuda.synchronize(device)
            order = modes.copy()
            rng.shuffle(order)
            reference = None
            if iteration == 0:
                values = torch.empty((args.topk, 576), dtype=torch.bfloat16)
                local_mask = selected < tokens // 2
                values[local_mask] = local[selected[local_mask].long(), 0].cpu()
                values[~local_mask] = host[host_ids, 0]
                values = values.to(device).float()
                scores = (
                    query[0].float() @ values[:, :512].T
                    + rope[0].float() @ values[:, 512:].T
                ) / 16
                reference = (scores.softmax(-1) @ values[:, :512]).unsqueeze(0)
            for mode in order:
                captured = graphs[mode][fraction if mode == "copy" else 0.0]
                eviction.fill_(iteration % 256)
                torch.cuda.synchronize(device)
                wall_start = time.perf_counter_ns()
                if mode == "copy":
                    pack_start = time.perf_counter_ns()
                    if rows:
                        torch.index_select(host, 0, host_ids, out=packed[:rows])
                    packing[fraction].append(
                        (time.perf_counter_ns() - pack_start) / 1000 if rows else 0.0
                    )
                cold_us = replay(captured["single"], 1)
                samples[fraction][mode]["serial_wall"].append(
                    (time.perf_counter_ns() - wall_start) / 1000
                )
                samples[fraction][mode]["cold_single"].append(cold_us)
                if reference is not None:
                    torch.testing.assert_close(
                        output.float(), reference, rtol=1e-2, atol=5e-3
                    )
                    errors[fraction][mode] = (
                        (output.float() - reference).abs().max().item()
                    )
                replay(captured["hot"], args.repeats)
                profile = (
                    iteration == 0
                    and fraction == args.profile_fraction
                    and mode == args.profile_mode
                )
                if profile:
                    torch.cuda.cudart().cudaProfilerStart()
                samples[fraction][mode]["gpu_cycle"].append(
                    replay(captured["hot"], args.repeats)
                )
                if profile:
                    torch.cuda.cudart().cudaProfilerStop()
        if (iteration + 1) % 10 == 0 or iteration + 1 == args.rounds:
            print(
                f"layout={args.layout} context={tokens} rounds={iteration + 1}/{args.rounds}",
                flush=True,
            )
    result = dict(
        context_tokens=tokens,
        residency=allocation.residency,
        gpu_before=before,
        gpu_after=gpu_snapshot(),
        cases=[],
    )
    for fraction in args.fractions:
        case = dict(
            host_fraction_requested=fraction,
            host_rows=round(fraction * args.topk),
            host_fraction_actual=round(fraction * args.topk) / args.topk,
            host_bytes=round(fraction * args.topk) * 1152,
            cpu_pack=summarize(packing[fraction]),
            correctness_max_abs_error=errors[fraction],
        )
        for mode in samples[fraction]:
            case[mode] = {
                metric: summarize(values)
                for metric, values in samples[fraction][mode].items()
            }
        result["cases"].append(case)
        print(
            f"context={tokens} host={fraction:.0%} copy={case['copy']['gpu_cycle']['median_us']:.3f} direct={case['direct']['gpu_cycle']['median_us']:.3f} overlap={case['overlap']['gpu_cycle']['median_us']:.3f}"
            + (
                f" previous={case['previous']['gpu_cycle']['median_us']:.3f}"
                if "previous" in case
                else ""
            ),
            flush=True,
        )
    del graphs, wrapper, data, kwargs, direct_wrapper, copy_wrapper
    del cache, local, host, alias_cache, staged, packed, query, rope, output
    del direct_indices, copy_indices, workspace, copy_workspace, eviction
    gc.collect()
    torch.cuda.synchronize(device)
    alias.close()
    allocation.close()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[131072, 262144, 524288, 1048576]
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[i / 100 for i in range(0, 41, 5)] + [0.6, 0.8, 1.0],
    )
    parser.add_argument(
        "--layout", choices=["random", "head", "tail"], default="random"
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--index-sets", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-fraction", type=float, default=None)
    parser.add_argument(
        "--profile-mode", choices=["copy", "direct", "overlap"], default="copy"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-module", type=Path)
    parser.add_argument(
        "--host-kernel", choices=("staged", "private"), default="staged"
    )
    parser.add_argument(
        "--private-chunks", type=int, choices=(28, 32, 40, 48, 60), default=28
    )
    parser.add_argument("--private-pipeline", action="store_true")
    parser.add_argument("--private-register", action="store_true")
    args = parser.parse_args()
    if args.private_register and not args.private_pipeline:
        raise ValueError("Register reads require --private-pipeline")
    if args.host_kernel == "private":
        from flashinfer.jit import gen_batch_mla_module
        from flashinfer.mla import _core

        @functools.cache
        def private_module(backend, *module_args, host_prefetch=False):
            specification = gen_batch_mla_module(
                backend, *module_args, host_prefetch=host_prefetch
            )
            if host_prefetch:
                specification.name += (
                    f"_private_shared_{args.private_chunks}_{int(args.private_pipeline)}"
                    + ("_register" if args.private_register else "")
                )
                specification.extra_cuda_cflags.extend(
                    [
                        "-DFLASHINFER_MLA_HOST_PRIVATE_SHARED",
                        f"-DFLASHINFER_MLA_HOST_PRIVATE_CHUNKS={args.private_chunks}",
                    ]
                )
                if args.private_pipeline:
                    specification.extra_cuda_cflags.append(
                        "-DFLASHINFER_MLA_HOST_PRIVATE_PIPELINE"
                    )
                if args.private_register:
                    specification.extra_cuda_cflags.append(
                        "-DFLASHINFER_MLA_HOST_PRIVATE_REGISTER"
                    )
            return specification.build_and_load()

        _core.get_batch_mla_module = private_module
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.topk != 2048 or 0 not in args.fractions:
        raise ValueError("This comparison requires top-k 2048 and a 0% baseline")
    os.sched_setaffinity(0, set(range(32)) | set(range(64, 96)))
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    result = dict(
        configuration=vars(args)
        | {
            "output": str(args.output),
            "baseline_module": str(args.baseline_module)
            if args.baseline_module
            else None,
        },
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "temp/bench_glm52_host_comparison.py",
                "include/flashinfer/attention/mla.cuh",
                "csrc/batch_mla_run.cu",
                "flashinfer/mla/_core.py",
            )
        },
        methods={
            "copy": "CPU packing into pinned buffer, then actual H2D DMA and native attention in one stream; local HBM is aliased without gathering",
            "direct": "Native attention directly reads mapped Host DRAM",
            "overlap": "Fused attention overlaps in-kernel Host reads with local computation",
        },
        timing={
            "gpu_cycle": "GPU events inside CUDA graph; every iteration includes its H2D DMA for copy and attention; CPU packing excluded",
            "cold_single": "One GPU cycle after 64 MiB L2 eviction",
            "serial_wall": "CPU wall time for one serialized cycle, including CPU packing for copy and final synchronization",
            "cpu_pack": "CPU gather into the pinned staging buffer",
        },
        exclusions=[
            "top-k selection",
            "plan",
            "index remapping/update",
            "initialization",
            "correctness reference",
        ],
        results=[],
    )
    if args.baseline_module:
        result["baseline_module_sha256"] = hashlib.sha256(
            args.baseline_module.read_bytes()
        ).hexdigest()
        result["methods"]["previous"] = (
            "Frozen previous fused Host-overlap implementation"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.contexts:
        result["results"].append(benchmark_context(args, tokens))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
