# Purpose: Benchmark sparse MLA decode with direct NVLink reads from remote HBM.
# Usage: PYTHONPATH=. python temp/bench_glm52_sparse_decode_p2p.py --output temp/glm52_sparse_decode_p2p.json

import argparse
import ctypes
import gc
import json
import math
import pathlib
import random
import re
import statistics
import subprocess
import time

import torch

try:
    from bench_glm52_decode import BatchMLAPagedAttentionWrapper
except ModuleNotFoundError as error:
    if error.name != "tvm_ffi":
        raise
    BatchMLAPagedAttentionWrapper = None


class MemoryLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compression", ctypes.c_ubyte),
        ("rdma", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class AllocationProperties(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle_types", ctypes.c_int),
        ("location", MemoryLocation),
        ("metadata", ctypes.c_void_p),
        ("flags", AllocationFlags),
    ]


class AccessDescriptor(ctypes.Structure):
    _fields_ = [("location", MemoryLocation), ("flags", ctypes.c_int)]


class DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [
    ("tensor", DLTensor),
    ("context", ctypes.c_void_p),
    ("deleter", DLDeleter),
]
_dlpack_owners = {}


@DLDeleter
def release_dlpack(pointer):
    _dlpack_owners.pop(ctypes.addressof(pointer.contents), None)


_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_new_capsule.restype = ctypes.py_object


class MixedHBM:
    """Map local and remote physical allocations into one CUDA virtual range."""

    def __init__(self, tokens, local, remote):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.address = ctypes.c_uint64()
        self.handles = []
        self.mappings = []
        self.tokens = tokens
        self.local_tokens = tokens // 2
        self.local = local
        self.remote = remote
        self.bytes_per_token = 576 * 2
        self.nbytes = tokens * self.bytes_per_token
        self.residency = []
        self.closed = False
        self._call("cuInit", ctypes.c_uint(0))
        devices = []
        for ordinal in (local, remote):
            device = ctypes.c_int()
            self._call("cuDeviceGet", ctypes.byref(device), ctypes.c_int(ordinal))
            devices.append(device.value)
        props = []
        granularities = []
        for device in devices:
            prop = AllocationProperties()
            prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED: device physical memory.
            prop.location = MemoryLocation(1, device)
            granularity = ctypes.c_size_t()
            self._call(
                "cuMemGetAllocationGranularity",
                ctypes.byref(granularity),
                ctypes.byref(prop),
                ctypes.c_int(0),
            )
            props.append(prop)
            granularities.append(granularity.value)
        chunk_bytes = self.local_tokens * self.bytes_per_token
        if any(chunk_bytes % granularity for granularity in granularities):
            raise ValueError("Each physical half must satisfy VMM granularity")
        self._call(
            "cuMemAddressReserve",
            ctypes.byref(self.address),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_size_t(max(granularities)),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0),
        )
        for part, prop in enumerate(props):
            handle = ctypes.c_uint64()
            self._call(
                "cuMemCreate",
                ctypes.byref(handle),
                ctypes.c_size_t(chunk_bytes),
                ctypes.byref(prop),
                ctypes.c_uint64(0),
            )
            self.handles.append(handle)
            address = self.address.value + part * chunk_bytes
            self._call(
                "cuMemMap",
                ctypes.c_uint64(address),
                ctypes.c_size_t(chunk_bytes),
                ctypes.c_size_t(0),
                handle,
                ctypes.c_uint64(0),
            )
            self.mappings.append((address, chunk_bytes))
            actual = AllocationProperties()
            self._call(
                "cuMemGetAllocationPropertiesFromHandle", ctypes.byref(actual), handle
            )
            assert actual.type == 1 and actual.location.type == 1
            assert actual.location.id == prop.location.id
            self.residency.append(
                {"owner_device": actual.location.id, "bytes": chunk_bytes}
            )
        access = (AccessDescriptor * 2)(
            *(AccessDescriptor(MemoryLocation(1, device), 3) for device in devices)
        )
        self._call(
            "cuMemSetAccess",
            self.address,
            ctypes.c_size_t(self.nbytes),
            access,
            ctypes.c_size_t(2),
        )

    def _call(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            error = ctypes.c_char_p()
            self.driver.cuGetErrorString(ctypes.c_int(status), ctypes.byref(error))
            raise RuntimeError(f"{name}: {error.value.decode()} ({status})")

    def tensor(self, device, start=0, tokens=None):
        if tokens is None:
            tokens = self.tokens - start
        shape = (ctypes.c_int64 * 3)(tokens, 1, 576)
        managed = DLManagedTensor()
        managed.tensor = DLTensor(
            self.address.value + start * self.bytes_per_token,
            DLDevice(2, device),  # kDLCUDA.
            3,
            DLDataType(4, 16, 1),  # kDLBfloat, BF16.
            shape,
            None,
            0,
        )
        managed.deleter = release_dlpack
        pointer = ctypes.addressof(managed)
        # Keep the allocation and DLPack structures alive until Tensor destruction.
        _dlpack_owners[pointer] = (managed, shape, self)
        capsule = _new_capsule(pointer, b"dltensor", None)
        return torch.utils.dlpack.from_dlpack(capsule)

    def close(self):
        if self.closed:
            return
        for address, nbytes in self.mappings:
            self._call("cuMemUnmap", ctypes.c_uint64(address), ctypes.c_size_t(nbytes))
        for handle in self.handles:
            self._call("cuMemRelease", handle)
        if self.address.value:
            self._call(
                "cuMemAddressFree", self.address, ctypes.c_size_t(self.nbytes)
            )
        self.closed = True


def gpu_snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        text=True,
    ).strip()


def nvlink_counters(device):
    output = subprocess.check_output(
        ["nvidia-smi", "nvlink", "-i", str(device), "-gt", "d"], text=True
    )
    totals = {"tx_bytes": 0, "rx_bytes": 0}
    for direction, kib in re.findall(r"Data (Tx|Rx): (\d+) KiB", output):
        totals[f"{direction.lower()}_bytes"] += int(kib) * 1024
    if not any(totals.values()):
        raise RuntimeError("NVLink traffic counters are unavailable")
    return totals


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int((len(ordered) - 1) * 0.1)],
        "p90_us": ordered[int((len(ordered) - 1) * 0.9)],
        "samples_us": samples,
    }


def capture_attention(wrapper, q_nope, q_pe, ckv, kpe, out, repeats):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    # Initialize event handles outside capture.
    start.record()
    end.record()
    end.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(repeats):
            wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
        end.record()
    return graph, start, end


def make_indices(tokens, topk, fractions, seed, layout):
    rng = random.Random(seed)
    half = tokens // 2
    local = rng.sample(range(half), topk)
    remote = rng.sample(range(half, tokens), topk)
    if layout == "random":
        replacement_order = rng.sample(range(topk), topk)
    elif layout == "head":
        replacement_order = list(range(topk))
    elif layout == "tail":
        replacement_order = list(reversed(range(topk)))
    else:
        raise ValueError(f"Unknown layout: {layout}")
    result = {}
    for fraction in fractions:
        count = round(fraction * topk)
        selected = local.copy()
        for position in replacement_order[:count]:
            selected[position] = remote[position]
        assert sum(token >= half for token in selected) == count
        assert len(set(selected)) == topk
        result[fraction] = torch.tensor(selected, dtype=torch.int32)
    return result


def benchmark_context(args, tokens):
    local = torch.device("cuda", args.local)
    remote = torch.device("cuda", args.remote)
    torch.cuda.set_device(local)
    torch.cuda.synchronize(local)
    torch.cuda.synchronize(remote)
    torch.cuda.empty_cache()
    allocation = MixedHBM(tokens, args.local, args.remote)
    cache = allocation.tensor(args.local)
    local_cache = allocation.tensor(args.local, 0, tokens // 2)
    remote_cache = allocation.tensor(args.remote, tokens // 2, tokens // 2)
    # Initialize each physical half with kernels running on its owner GPU.
    with torch.cuda.device(local):
        torch.randn(local_cache.shape, out=local_cache)
    with torch.cuda.device(remote):
        torch.randn(remote_cache.shape, out=remote_cache)
    torch.cuda.synchronize(local)
    torch.cuda.synchronize(remote)
    del local_cache, remote_cache
    ckv, kpe = cache[..., :512], cache[..., 512:]
    q_nope = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=local)
    q_pe = torch.randn((1, 64, 64), dtype=torch.bfloat16, device=local)
    out = torch.empty_like(q_nope)
    indices = torch.empty(args.topk, dtype=torch.int32, device=local)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=local)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.tensor([0, 1], dtype=torch.int32, device=local),
        torch.tensor([0, args.topk], dtype=torch.int32, device=local),
        indices,
        torch.tensor([args.topk], dtype=torch.int32, device=local),
        num_heads=64,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=False,
        sm_scale=1 / math.sqrt(256),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    index_sets = [
        make_indices(tokens, args.topk, args.fractions, args.seed + i, args.layout)
        for i in range(args.index_sets)
    ]
    indices.copy_(index_sets[0][0.0])
    for _ in range(3):
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    torch.cuda.synchronize(local)
    cold_graph, cold_start, cold_end = capture_attention(
        wrapper, q_nope, q_pe, ckv, kpe, out, 1
    )
    hot_graph, hot_start, hot_end = capture_attention(
        wrapper, q_nope, q_pe, ckv, kpe, out, args.hot_repeats
    )
    # Both A100 GPUs have 40 MiB L2. A read/write sweep of 64 MiB evicts KV data.
    local_flush = torch.zeros(args.flush_mib * 2**18, device=local)
    remote_flush = torch.zeros(args.flush_mib * 2**18, device=remote)
    remote_done = torch.cuda.Event()
    samples = {fraction: {"cold": [], "hot": []} for fraction in args.fractions}
    errors = {}
    result = {
        "context_tokens": tokens,
        "residency": allocation.residency,
        "kv_cache_bytes": allocation.nbytes,
        "cache_stride_elements": list(cache.stride()),
        "gpu_before": gpu_snapshot(),
    }
    order_rng = random.Random(args.seed + tokens)
    for iteration in range(args.rounds):
        fractions = args.fractions.copy()
        order_rng.shuffle(fractions)
        for fraction in fractions:
            selected = index_sets[iteration % len(index_sets)][fraction]
            # This copies only 8 KiB of page-table metadata, never KV cache data.
            indices.copy_(selected)
            with torch.cuda.device(remote):
                remote_flush.add_(1)
                remote_done.record()
            local_flush.add_(1)
            torch.cuda.current_stream(local).wait_event(remote_done)
            cold_graph.replay()
            cold_end.synchronize()
            samples[fraction]["cold"].append(cold_start.elapsed_time(cold_end) * 1000)
            if iteration == 0:
                # Reference gathers are validation kernels outside measured graphs.
                selected_cuda = selected.to(local, dtype=torch.int64)
                selected_ckv = ckv[selected_cuda, 0].float()
                selected_kpe = kpe[selected_cuda, 0].float()
                scores = (
                    q_nope[0].float() @ selected_ckv.T
                    + q_pe[0].float() @ selected_kpe.T
                ) / math.sqrt(256)
                reference = (scores.softmax(-1) @ selected_ckv).unsqueeze(0)
                torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=5e-3)
                errors[fraction] = (out.float() - reference).abs().max().item()
                del selected_cuda, selected_ckv, selected_kpe, scores, reference
            hot_graph.replay()
            hot_end.synchronize()
            samples[fraction]["hot"].append(
                hot_start.elapsed_time(hot_end) * 1000 / args.hot_repeats
            )
        if (iteration + 1) % 10 == 0:
            print(f"context={tokens} rounds={iteration + 1}/{args.rounds}", flush=True)
    result["gpu_after"] = gpu_snapshot()
    if args.verify_nvlink:
        result["nvlink_validation"] = []
        for fraction in args.fractions:
            indices.copy_(index_sets[0][fraction])
            hot_graph.replay()
            hot_end.synchronize()
            before = nvlink_counters(args.local)
            hot_graph.replay()
            hot_end.synchronize()
            after = nvlink_counters(args.local)
            traffic = {key: after[key] - before[key] for key in before}
            traffic.update(
                remote_fraction=fraction, attention_calls=args.hot_repeats
            )
            result["nvlink_validation"].append(traffic)
            print(f"nvlink_validation={traffic}", flush=True)
    result["cases"] = []
    baseline = samples[0.0]
    for fraction in args.fractions:
        case = {
            "remote_fraction_requested": fraction,
            "remote_tokens": round(fraction * args.topk),
            "remote_fraction_actual": round(fraction * args.topk) / args.topk,
            "correctness_max_abs_error": errors[fraction],
        }
        for mode in ("cold", "hot"):
            case[mode] = summarize(samples[fraction][mode])
            case[mode]["relative_to_local"] = (
                case[mode]["median_us"] / statistics.median(baseline[mode])
            )
            case[mode]["paired_relative_median"] = statistics.median(
                value / control
                for value, control in zip(samples[fraction][mode], baseline[mode])
            )
        result["cases"].append(case)
        print(
            f"context={tokens} remote={fraction:.0%} "
            f"cold_us={case['cold']['median_us']:.3f} "
            f"cold_ratio={case['cold']['relative_to_local']:.3f} "
            f"hot_us={case['hot']['median_us']:.3f} "
            f"hot_ratio={case['hot']['relative_to_local']:.3f} "
            f"max_error={errors[fraction]:.6f}",
            flush=True,
        )
    torch.cuda.synchronize(local)
    torch.cuda.synchronize(remote)
    del cold_graph, hot_graph, wrapper, ckv, kpe, cache
    gc.collect()
    allocation.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[131072, 262144, 524288, 1048576])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--layout", choices=("random", "head", "tail"), default="random")
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--remote", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--index-sets", type=int, default=8)
    parser.add_argument("--hot-repeats", type=int, default=100)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-nvlink", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("temp/glm52_sparse_decode_p2p.json"))
    args = parser.parse_args()
    if 0 not in args.fractions or any(not 0 <= fraction <= 1 for fraction in args.fractions):
        raise ValueError("Fractions must include zero and lie in [0, 1]")
    if args.local == args.remote or not torch.cuda.can_device_access_peer(args.local, args.remote):
        raise RuntimeError("Two GPUs with CUDA peer access are required")
    if any(tokens // 2 < args.topk for tokens in args.contexts):
        raise ValueError("Each physical half must contain at least topk tokens")
    if args.rounds < 1 or args.index_sets < 1 or args.hot_repeats < 1 or args.flush_mib < 64:
        raise ValueError("Positive counts and at least 64 MiB cache sweeps are required")
    torch.manual_seed(args.seed)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "local_device": args.local,
        "remote_device": args.remote,
        "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "configuration": vars(args) | {"output": str(args.output)},
        "attention": {
            "wrapper": "BatchMLAPagedAttentionWrapper",
            "backend": "fa2",
            "batch_size": 1,
            "query_tokens": 1,
            "heads": 64,
            "compressed_rank": 512,
            "rope_dim": 64,
            "dtype": "bfloat16",
            "page_size": 1,
            "kv_transfer": "none; direct P2P loads through CUDA VMM",
            "timing": "CUDA events inside captured graphs; attention kernels only",
            "cold": "64 MiB read/write sweep on each owner GPU before one attention call",
            "hot": "Repeated calls using the same selected indices",
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(result["topology"], flush=True)
    for tokens in args.contexts:
        result["results"].append(benchmark_context(args, tokens))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
