# Purpose: Benchmark MLA attention directly reading a cache split across HBM and pinned Host DRAM.
# Usage: PYTHONPATH=. python temp/bench_glm52_sparse_decode_hbm_host_direct.py --device 1 --output temp/glm52_hbm_host_direct_random.json

import argparse
import ctypes
import gc
import json
import math
import pathlib
import random
import statistics
import time

import torch

from bench_glm52_decode import BatchMLAPagedAttentionWrapper
from bench_glm52_sparse_decode_p2p import (
    AccessDescriptor,
    AllocationProperties,
    MemoryLocation,
    MixedHBM,
    gpu_snapshot,
    make_indices,
    summarize,
)


class MixedHBMHost(MixedHBM):
    """Map device HBM and fixed Host NUMA memory into one CUDA address range."""

    def __init__(self, tokens, device):
        torch.cuda.set_device(device)
        torch.empty(1, device=f"cuda:{device}")
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.address = ctypes.c_uint64()
        self.handles = []
        self.mappings = []
        self.tokens = tokens
        self.local_tokens = tokens // 2
        self.local = device
        self.bytes_per_token = 576 * 2
        self.nbytes = tokens * self.bytes_per_token
        self.residency = []
        self.closed = False
        self._call("cuInit", ctypes.c_uint(0))
        numa = ctypes.c_int()
        self._call("cuDeviceGetAttribute", ctypes.byref(numa), ctypes.c_int(134), ctypes.c_int(device))
        self.host_numa = max(0, numa.value)
        supported = ctypes.c_int()
        self._call("cuDeviceGetAttribute", ctypes.byref(supported), ctypes.c_int(141), ctypes.c_int(device))
        if not supported.value:
            raise RuntimeError("Host NUMA VMM is required for an unchanged MLA kernel with mixed residency")
        props = []
        granularities = []
        for location in (MemoryLocation(1, device), MemoryLocation(3, self.host_numa)):
            prop = AllocationProperties()
            prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED; physical pages cannot migrate.
            prop.location = location
            granularity = ctypes.c_size_t()
            self._call("cuMemGetAllocationGranularity", ctypes.byref(granularity), ctypes.byref(prop), ctypes.c_int(0))
            props.append(prop)
            granularities.append(granularity.value)
        half_bytes = self.local_tokens * self.bytes_per_token
        if any(half_bytes % granularity for granularity in granularities):
            raise ValueError("Each cache half must satisfy its VMM allocation granularity")
        self._call("cuMemAddressReserve", ctypes.byref(self.address), ctypes.c_size_t(self.nbytes), ctypes.c_size_t(max(granularities)), ctypes.c_uint64(0), ctypes.c_uint64(0))
        try:
            for part, prop in enumerate(props):
                handle = ctypes.c_uint64()
                self._call("cuMemCreate", ctypes.byref(handle), ctypes.c_size_t(half_bytes), ctypes.byref(prop), ctypes.c_uint64(0))
                self.handles.append(handle)
                address = self.address.value + part * half_bytes
                self._call("cuMemMap", ctypes.c_uint64(address), ctypes.c_size_t(half_bytes), ctypes.c_size_t(0), handle, ctypes.c_uint64(0))
                self.mappings.append((address, half_bytes))
                access = [AccessDescriptor(MemoryLocation(1, device), 3)]
                if part == 1:
                    access.append(AccessDescriptor(MemoryLocation(3, self.host_numa), 3))
                descriptors = (AccessDescriptor * len(access))(*access)
                self._call("cuMemSetAccess", ctypes.c_uint64(address), ctypes.c_size_t(half_bytes), descriptors, ctypes.c_size_t(len(access)))
                actual = AllocationProperties()
                self._call("cuMemGetAllocationPropertiesFromHandle", ctypes.byref(actual), handle)
                assert actual.type == prop.type and actual.location.type == prop.location.type
                assert actual.location.id == prop.location.id
                self.residency.append({
                    "tier": "hbm" if part == 0 else "pinned_host_dram",
                    "allocation_type": actual.type,
                    "location_type": actual.location.type,
                    "location_id": actual.location.id,
                    "address": hex(address),
                    "bytes": half_bytes,
                })
        except Exception:
            self.close()
            raise

    def host_tensor(self):
        address = self.address.value + self.local_tokens * self.bytes_per_token
        buffer = (ctypes.c_uint16 * (self.local_tokens * 576)).from_address(address)
        return torch.frombuffer(buffer, dtype=torch.bfloat16).view(self.local_tokens, 1, 576)


def capture_attention(wrapper, q_nope, q_pe, ckv, kpe, out, repeats):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    return graph


def time_gpu(function, repeats, start, end):
    start.record()
    function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def benchmark_context(args, tokens):
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    before = gpu_snapshot()
    allocation = MixedHBMHost(tokens, args.device)
    cache = allocation.tensor(args.device)
    local_cache = cache[:tokens // 2]
    host_cpu = allocation.host_tensor()
    local_cache.normal_()
    host_cpu.normal_()
    assert not host_cpu.is_cuda
    ckv, kpe = cache[..., :512], cache[..., 512:]
    q_nope = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((1, 64, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q_nope)
    indices = torch.empty(args.topk, dtype=torch.int32, device=device)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0, args.topk], dtype=torch.int32, device=device),
        indices,
        torch.tensor([args.topk], dtype=torch.int32, device=device),
        num_heads=64, head_dim_ckv=512, head_dim_kpe=64, page_size=1,
        causal=False, sm_scale=1 / math.sqrt(256),
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    assert wrapper._kv_indices_buf.data_ptr() == indices.data_ptr()
    index_sets = [make_indices(tokens, args.topk, args.fractions, args.seed + i, args.layout) for i in range(args.index_sets)]
    indices.copy_(index_sets[0][0.0])
    for _ in range(5):
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    torch.cuda.synchronize(device)
    single_graph = capture_attention(wrapper, q_nope, q_pe, ckv, kpe, out, 1)
    hot_graph = capture_attention(wrapper, q_nope, q_pe, ckv, kpe, out, args.hot_repeats)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    eviction = torch.empty(args.flush_mib * 2**20, dtype=torch.uint8, device=device)
    samples = {fraction: {mode: [] for mode in ("hot_graph", "cold_single_graph", "eager")} for fraction in args.fractions}
    errors = {}
    rng = random.Random(args.seed + tokens)
    for iteration in range(args.rounds):
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            selected = index_sets[iteration % len(index_sets)][fraction]
            host_count = int((selected >= tokens // 2).sum())
            assert host_count == round(fraction * args.topk)
            indices.copy_(selected)
            eviction.fill_(iteration % 256)
            torch.cuda.synchronize(device)
            samples[fraction]["cold_single_graph"].append(time_gpu(single_graph.replay, 1, start, end))
            if iteration == 0:
                local_mask = selected < tokens // 2
                rows = torch.empty((args.topk, 576), dtype=torch.bfloat16)
                if local_mask.any():
                    rows[local_mask] = local_cache[selected[local_mask].long(), 0].cpu()
                if (~local_mask).any():
                    rows[~local_mask] = host_cpu[(selected[~local_mask] - tokens // 2).long(), 0]
                rows_gpu = rows.to(device).float()
                scores = (q_nope[0].float() @ rows_gpu[:, :512].T + q_pe[0].float() @ rows_gpu[:, 512:].T) / math.sqrt(256)
                reference = (scores.softmax(-1) @ rows_gpu[:, :512]).unsqueeze(0)
                torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=5e-3)
                errors[fraction] = (out.float() - reference).abs().max().item()
            hot_graph.replay()
            torch.cuda.synchronize(device)
            profile = iteration == 0 and fraction == args.profile_fraction
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
            samples[fraction]["hot_graph"].append(time_gpu(hot_graph.replay, args.hot_repeats, start, end))
            if profile:
                torch.cuda.cudart().cudaProfilerStop()

            def eager():
                for _ in range(args.eager_repeats):
                    wrapper.run(q_nope, q_pe, ckv, kpe, out=out)

            samples[fraction]["eager"].append(time_gpu(eager, args.eager_repeats, start, end))
        if (iteration + 1) % 10 == 0:
            print(f"layout={args.layout} context={tokens} rounds={iteration + 1}/{args.rounds}", flush=True)
    result = {"context_tokens": tokens, "kv_cache_bytes": tokens * 1152, "residency": allocation.residency, "gpu_before": before, "gpu_after": gpu_snapshot(), "cases": []}
    for fraction in args.fractions:
        host_tokens = round(fraction * args.topk)
        case = {"host_fraction_requested": fraction, "host_tokens": host_tokens, "host_fraction_actual": host_tokens / args.topk, "host_kv_bytes": host_tokens * 1152, "correctness_max_abs_error": errors[fraction]}
        for mode, values in samples[fraction].items():
            case[mode] = summarize(values)
            case[mode]["relative_to_hbm"] = statistics.median(values) / statistics.median(samples[0.0][mode])
        result["cases"].append(case)
        print(f"layout={args.layout} context={tokens} host={fraction:.0%} graph_us={case['hot_graph']['median_us']:.3f} cold_us={case['cold_single_graph']['median_us']:.3f} eager_us={case['eager']['median_us']:.3f}", flush=True)
    del single_graph, hot_graph, wrapper, cache, local_cache, host_cpu, ckv, kpe, q_nope, q_pe, out, workspace, eviction
    gc.collect()
    torch.cuda.synchronize(device)
    allocation.close()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[131072, 262144, 524288, 1048576])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, .1, .15, .2, .3, .4, .6, .8, 1])
    parser.add_argument("--layout", choices=("random", "head", "tail"), default="random")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--index-sets", type=int, default=8)
    parser.add_argument("--hot-repeats", type=int, default=100)
    parser.add_argument("--eager-repeats", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--profile-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("temp/glm52_hbm_host_direct_random.json"))
    args = parser.parse_args()
    if 0 not in args.fractions or any(not 0 <= f <= 1 for f in args.fractions):
        raise ValueError("fractions must include 0 and lie in [0, 1]")
    torch.manual_seed(args.seed)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(args.device),
        "configuration": vars(args) | {"output": str(args.output)},
        "access": "First cache half in HBM, second half in pinned Host NUMA DRAM; unchanged MLA kernel directly reads both via one VMM range",
        "timing": "hot_graph: 100 attention calls per CUDA Graph, GPU events around replay; eager: Python launches including GPU idle gaps; cold_single_graph: one call after L2 eviction, including graph launch gap",
        "exclusions": ["top-k selection", "plan", "index metadata update", "correctness reference", "initialization"],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Preserving existing benchmark artifact: {args.output}")
    for tokens in args.contexts:
        result["results"].append(benchmark_context(args, tokens))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
