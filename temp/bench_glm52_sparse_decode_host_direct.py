# Purpose: Benchmark MLA sparse decode with KV cache directly in mapped Host DRAM.
# Usage: PYTHONPATH=. python temp/bench_glm52_sparse_decode_host_direct.py --output temp/glm52_sparse_decode_host_direct.json

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

from bench_glm52_decode import BatchMLAPagedAttentionWrapper
from bench_glm52_sparse_decode_p2p import make_indices
from mapped_host_tensor_ext import allocate


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem", "--format=csv"],
        text=True,
    ).strip()


def summarize(values):
    values = sorted(values)
    return {"median_us": statistics.median(values), "p10_us": values[int(.1 * (len(values) - 1))], "p90_us": values[int(.9 * (len(values) - 1))], "samples_us": values}


def benchmark_context(args, tokens):
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    # The cache is exposed as a CUDA tensor whose data pointer is a mapped Host pointer.
    cache = allocate((tokens, 576), args.device)
    cache.normal_()
    q_nope = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((1, 64, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q_nope)
    workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, args.topk], dtype=torch.int32, device=device)
    indices = torch.empty(args.topk, dtype=torch.int32, device=device)
    kv_len = torch.tensor([args.topk], dtype=torch.int32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(qo_indptr, kv_indptr, indices, kv_len, num_heads=64, head_dim_ckv=512, head_dim_kpe=64, page_size=1, causal=False, sm_scale=1 / math.sqrt(256), q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    ckv = cache[:, :512].view(tokens, 1, 512)
    kpe = cache[:, 512:].view(tokens, 1, 64)
    index_sets = [make_indices(tokens, args.topk, args.fractions, args.seed + i, args.layout) for i in range(args.index_sets)]
    samples = {f: [] for f in args.fractions}
    errors = {}
    rng = random.Random(args.seed + tokens)
    for iteration in range(args.rounds):
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            selected = index_sets[iteration % len(index_sets)][fraction]
            indices.copy_(selected)
            for _ in range(3):
                wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
            end.record()
            end.synchronize()
            samples[fraction].append(start.elapsed_time(end) * 1000 / args.repeats)
            if iteration == 0:
                rows = cache[selected.long()]
                scores = (q_nope[0].float() @ rows[:, :512].float().T + q_pe[0].float() @ rows[:, 512:].float().T) / math.sqrt(256)
                reference = (scores.softmax(-1) @ rows[:, :512].float()).unsqueeze(0)
                torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=5e-3)
                errors[fraction] = (out.float() - reference).abs().max().item()
        if (iteration + 1) % 10 == 0:
            print(f"context={tokens} rounds={iteration + 1}/{args.rounds}", flush=True)
    baseline = statistics.median(samples[0.0])
    result = {"context_tokens": tokens, "kv_cache_bytes": tokens * 576 * 2, "gpu_before": gpu_snapshot(), "gpu_after": gpu_snapshot(), "cases": []}
    for fraction in args.fractions:
        timing = summarize(samples[fraction])
        timing["relative_to_local"] = timing["median_us"] / baseline
        result["cases"].append({"remote_fraction_requested": fraction, "remote_tokens": round(fraction * args.topk), "correctness_max_abs_error": errors[fraction], "direct_attention": timing})
        print(f"context={tokens} host_fraction={fraction:.0%} direct_attention_us={timing['median_us']:.3f}", flush=True)
    del wrapper, cache, ckv, kpe, workspace, q_nope, q_pe, out
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("temp/glm52_sparse_decode_host_direct.json"))
    args = parser.parse_args()
    if 0 not in args.fractions or any(not 0 <= f <= 1 for f in args.fractions):
        raise ValueError("fractions must include 0 and lie in [0, 1]")
    torch.manual_seed(args.seed)
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "configuration": vars(args) | {"output": str(args.output)}, "access": "MLA directly reads CUDA tensor backed by cudaHostAllocMapped Host DRAM; no staging or memcpy", "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.contexts:
        result["results"].append(benchmark_context(args, tokens))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
