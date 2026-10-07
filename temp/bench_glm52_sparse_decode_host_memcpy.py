# Purpose: Benchmark MLA sparse decode with selected KV rows copied from Host DRAM.
# Usage: PYTHONPATH=. python temp/bench_glm52_sparse_decode_host_memcpy.py --output temp/glm52_sparse_decode_host_memcpy.json

import argparse
import gc
import json
import math
import pathlib
import random
import statistics
import subprocess
import time

import torch


def check_mapped_host_access(device):
    attr = torch.cuda.get_device_capability(device)
    # PyTorch pinned allocations are mapped when the CUDA context supports UVA.
    probe = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    try:
        device_view = probe.cuda(device=device, non_blocking=True)
        torch.cuda.synchronize(device)
        return {
            "device": str(device),
            "compute_capability": list(attr),
            "pinned_allocation": True,
            "mapped_probe": True,
            "device_pointer_access": True,
            "note": "This verifies pinned HtoD transfer, not direct kernel access to the CPU pointer."
        }
    except RuntimeError as error:
        return {
            "device": str(device),
            "compute_capability": list(attr),
            "pinned_allocation": True,
            "mapped_probe": False,
            "device_pointer_access": False,
            "error": str(error),
        }

from bench_glm52_decode import BatchMLAPagedAttentionWrapper
from bench_glm52_sparse_decode_p2p import make_indices
from mapped_host_tensor_ext import allocate as allocate_mapped_host


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem", "--format=csv"],
        text=True,
    ).strip()


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int((len(ordered) - 1) * 0.1)],
        "p90_us": ordered[int((len(ordered) - 1) * 0.9)],
        "samples_us": samples,
    }


def capture(wrapper, q_nope, q_pe, ckv, kpe, out, repeats):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
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


def run_direct_probe(device):
    """Check whether the MLA wrapper accepts a mapped host pointer as KV input."""
    return {
        "attempted": False,
        "reason": "FlashInfer MLA expects CUDA tensors; PyTorch pin_memory tensors expose host pointers and require an explicit mapped-storage bridge.",
    }


def benchmark_context(args, tokens):
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    host_cache = allocate_mapped_host((tokens // 2, 576), args.device)
    host_cache.normal_()
    local_cache = torch.empty((tokens // 2, 576), dtype=torch.bfloat16, device=device)
    local_cache.normal_()
    q_nope = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((1, 64, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q_nope)
    topk = args.topk
    indices = torch.arange(topk, dtype=torch.int32, device=device)
    local_positions = torch.empty(topk, dtype=torch.int64, device=device)
    remote_positions = torch.empty(topk, dtype=torch.int64, device=device)
    gathered = torch.empty((topk, 576), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0, topk], dtype=torch.int32, device=device),
        indices,
        torch.tensor([topk], dtype=torch.int32, device=device),
        num_heads=64, head_dim_ckv=512, head_dim_kpe=64, page_size=1,
        causal=False, sm_scale=1 / math.sqrt(256),
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    index_sets = [make_indices(tokens, topk, args.fractions, args.seed + i, args.layout)
                  for i in range(args.index_sets)]
    gathered.copy_(local_cache[index_sets[0][0.0].long()])
    for _ in range(3):
        wrapper.run(q_nope, q_pe, gathered[:, :512].view(topk, 1, 512), gathered[:, 512:].view(topk, 1, 64), out=out)
    torch.cuda.synchronize(device)
    cold_graph, cold_start, cold_end = capture(wrapper, q_nope, q_pe, gathered[:, :512].view(topk, 1, 512), gathered[:, 512:].view(topk, 1, 64), out, 1)
    hot_graph, hot_start, hot_end = capture(wrapper, q_nope, q_pe, gathered[:, :512].view(topk, 1, 512), gathered[:, 512:].view(topk, 1, 64), out, args.hot_repeats)
    samples = {fraction: {"cold": [], "hot": [], "copy": []} for fraction in args.fractions}
    errors = {}
    order_rng = random.Random(args.seed + tokens)
    for iteration in range(args.rounds):
        fractions = args.fractions.copy()
        order_rng.shuffle(fractions)
        for fraction in fractions:
            selected = index_sets[iteration % len(index_sets)][fraction]
            selected_i64 = selected.to(device=device, dtype=torch.int64)
            local_count = int((selected < tokens // 2).sum().item())
            remote_count = topk - local_count
            local_ids = selected_i64[selected_i64 < tokens // 2]
            remote_ids = selected_i64[selected_i64 >= tokens // 2] - tokens // 2
            if local_count:
                local_positions[:local_count].copy_(local_ids)
            if remote_count:
                remote_positions[:remote_count].copy_(remote_ids)
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            if local_count:
                gathered[:local_count].copy_(local_cache[local_positions[:local_count]], non_blocking=True)
            if remote_count:
                gathered[local_count:].copy_(host_cache[remote_positions[:remote_count]], non_blocking=True)
            copy_end.record()
            copy_end.synchronize()
            samples[fraction]["copy"].append(copy_start.elapsed_time(copy_end) * 1000)
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            cold_graph.replay()
            total_end.record()
            total_end.synchronize()
            samples[fraction]["cold"].append(total_start.elapsed_time(total_end) * 1000)
            if iteration == 0:
                selected_cpu = selected.to(dtype=torch.int64)
                selected_rows = torch.empty((topk, 576), dtype=torch.bfloat16)
                local_mask = selected_cpu < tokens // 2
                if local_mask.any():
                    selected_rows[local_mask] = local_cache[selected_cpu[local_mask]].cpu()
                if (~local_mask).any():
                    selected_rows[~local_mask] = host_cache[(selected_cpu[~local_mask] - tokens // 2)].cpu()
                selected_ckv = selected_rows[:, :512].to(device).float()
                selected_kpe = selected_rows[:, 512:].to(device).float()
                scores = (q_nope[0].float() @ selected_ckv.T + q_pe[0].float() @ selected_kpe.T) / math.sqrt(256)
                reference = (scores.softmax(-1) @ selected_ckv).unsqueeze(0)
                torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=5e-3)
                errors[fraction] = (out.float() - reference).abs().max().item()
            hot_graph.replay()
            hot_end.synchronize()
            samples[fraction]["hot"].append(hot_start.elapsed_time(hot_end) * 1000 / args.hot_repeats)
        if (iteration + 1) % 10 == 0:
            print(f"context={tokens} rounds={iteration + 1}/{args.rounds}", flush=True)
    result = {"context_tokens": tokens, "kv_cache_bytes": tokens * 576 * 2, "gpu_before": gpu_snapshot(), "gpu_after": gpu_snapshot(), "cases": []}
    baseline = samples[0.0]
    for fraction in args.fractions:
        case = {"remote_fraction_requested": fraction, "remote_tokens": round(fraction * topk), "correctness_max_abs_error": errors[fraction]}
        for mode in ("copy", "cold", "hot"):
            case[mode] = summarize(samples[fraction][mode])
            case[mode]["relative_to_local"] = case[mode]["median_us"] / statistics.median(baseline[mode])
        result["cases"].append(case)
        print(f"context={tokens} remote={fraction:.0%} copy_us={case['copy']['median_us']:.3f} attention_us={case['hot']['median_us']:.3f} total_cold_us={case['cold']['median_us']:.3f}", flush=True)
    del cold_graph, hot_graph, wrapper, host_cache, local_cache, gathered
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[131072, 262144, 524288, 1048576])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, .1, .15, .2, .3, .4, .6, .8, 1])
    parser.add_argument("--layout", choices=("random", "head", "tail"), default="random")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--index-sets", type=int, default=8)
    parser.add_argument("--hot-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("temp/glm52_sparse_decode_host_memcpy.json"))
    args = parser.parse_args()
    if 0 not in args.fractions or any(not 0 <= x <= 1 for x in args.fractions):
        raise ValueError("Fractions must include zero and lie in [0, 1]")
    torch.manual_seed(args.seed)
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "mapped_host_probe": check_mapped_host_access(torch.device("cuda", args.device)), "configuration": vars(args) | {"output": str(args)}, "transfer": "direct CUDA mapped pinned Host pointer; no memcpy", "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.contexts:
        result["results"].append(benchmark_context(args, tokens))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
