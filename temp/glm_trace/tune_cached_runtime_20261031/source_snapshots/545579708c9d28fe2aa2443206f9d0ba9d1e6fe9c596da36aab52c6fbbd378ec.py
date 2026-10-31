# Purpose: Compare serial pinned-H2D copy and native direct Host reads for batched MTP sparse MLA decode.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_batch.py --device 3 --output temp/host_attention_batch/random.json
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_glm52_host_comparison import StagedHBM  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402
from bench_glm52_sparse_decode_p2p import gpu_snapshot, summarize  # noqa: E402
from flashinfer.mla import BatchMLAPagedAttentionWrapper  # noqa: E402

HEADS = 64
SM_SCALE = 1 / 16


def make_indices(requests, mtp, context, total_tokens, topk, fractions, seed, device):
    """Independent per-token top-k sets; request r owns local and Host halves of its context."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = requests * mtp
    half = context // 2
    owner = torch.arange(rows, device=device) // mtp
    local = torch.empty((rows, topk), dtype=torch.int64, device=device)
    host = torch.empty_like(local)
    for begin in range(0, rows, 64):
        end = min(rows, begin + 64)
        for out in (local, host):
            noise = torch.rand(
                (end - begin, half), generator=generator, device=device
            )
            out[begin:end] = noise.topk(topk, dim=1).indices
    local += owner[:, None] * half
    host += total_tokens // 2 + owner[:, None] * half
    order = torch.rand((rows, topk), generator=generator, device=device).argsort(1)
    result = {}
    for fraction in fractions:
        count = round(fraction * topk)
        selected = local.clone()
        positions = order[:, :count]
        selected.scatter_(1, positions, host.gather(1, positions))
        assert int((selected >= total_tokens // 2).sum()) == rows * count
        result[fraction] = selected.int()
    return result


def make_wrapper(rows, indices, workspace, topk):
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.arange(rows + 1, dtype=torch.int32),
        torch.arange(rows + 1, dtype=torch.int32) * topk,
        indices,
        torch.full((rows,), topk, dtype=torch.int32),
        HEADS,
        512,
        64,
        1,
        False,
        SM_SCALE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    assert wrapper._kv_indices_buf.data_ptr() == indices.data_ptr()
    return wrapper


def capture(wrapper, query, rope, cache, output, repeats, source=None, destination=None, rows=0):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(repeats):
            if rows:
                destination[:rows].copy_(source[:rows], non_blocking=True)
            wrapper.run(query, rope, cache[..., :512], cache[..., 512:], out=output)
        end.record()
    return graph, start, end


def replay(captured, repeats):
    graph, start, end = captured
    graph.replay()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def reference(cache, query, rope, selected, rows):
    expected = []
    for row in rows:
        values = cache[selected[row].long(), 0].float()
        scores = (
            query[row].float() @ values[:, :512].T + rope[row].float() @ values[:, 512:].T
        ) * SM_SCALE
        expected.append(scores.softmax(-1) @ values[:, :512])
    return torch.stack(expected)


def benchmark_batch(args, requests, allocation, cache, host, alias_cache, staged, packed):
    device = torch.device("cuda", args.device)
    tokens = allocation.tokens
    rows = requests * args.mtp
    topk = args.topk
    query = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((rows, HEADS, 64), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    direct_indices = torch.empty(rows * topk, dtype=torch.int32, device=device)
    copy_indices = torch.empty_like(direct_indices)
    workspace = torch.empty(args.workspace_mib * 2**20, dtype=torch.uint8, device=device)
    copy_workspace = torch.empty_like(workspace)
    direct_wrapper = make_wrapper(rows, direct_indices, workspace, topk)
    copy_wrapper = make_wrapper(rows, copy_indices, copy_workspace, topk)

    # Per index set and fraction: direct indices, unique Host rows, and remapped copy indices.
    prepared = []
    for set_id in range(args.index_sets):
        selections = make_indices(
            requests,
            args.mtp,
            args.context,
            tokens,
            topk,
            args.fractions,
            args.seed + 1000 * requests + set_id,
            device,
        )
        case_set = {}
        for fraction, selected in selections.items():
            flat = selected.view(-1)
            mask = flat >= tokens // 2
            unique, inverse = torch.unique(flat[mask], return_inverse=True)
            remapped = flat.clone()
            remapped[mask] = (tokens // 2 + inverse).int()
            case_set[fraction] = dict(
                selected=selected,
                remapped=remapped,
                host_ids=(unique - tokens // 2).cpu(),
                host_refs=int(mask.sum()),
            )
        prepared.append(case_set)
        del selections

    def run_direct():
        direct_wrapper.run(query, rope, cache[..., :512], cache[..., 512:], out=output)

    def run_copy(unique_rows):
        if unique_rows:
            staged[:unique_rows].copy_(packed[:unique_rows], non_blocking=True)
        copy_wrapper.run(query, rope, alias_cache[..., :512], alias_cache[..., 512:], out=output)

    # Warm up, estimate one call per case, and choose graph lengths near target_ms.
    first = prepared[0]
    direct_indices.copy_(first[0.0]["selected"].view(-1))
    copy_indices.copy_(first[0.0]["remapped"])
    for _ in range(3):
        run_direct()
        run_copy(0)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeats = {}
    for fraction in args.fractions:
        direct_indices.copy_(first[fraction]["selected"].view(-1))
        copy_indices.copy_(first[fraction]["remapped"])
        unique_rows = len(first[fraction]["host_ids"])
        for mode, function in (("direct", run_direct), ("copy", lambda: run_copy(unique_rows))):
            function()
            start.record()
            function()
            end.record()
            end.synchronize()
            estimate_us = start.elapsed_time(end) * 1000
            repeats[(mode, fraction)] = max(
                1, min(args.max_repeats, math.ceil(args.target_ms * 1000 / estimate_us))
            )

    # Copy graphs bake the unique row count, so capture one per index set.
    graphs = {}
    for fraction in args.fractions:
        graphs[("direct", fraction)] = capture(
            direct_wrapper, query, rope, cache, output, repeats[("direct", fraction)]
        )
        for set_id, case_set in enumerate(prepared):
            graphs[("copy", fraction, set_id)] = capture(
                copy_wrapper,
                query,
                rope,
                alias_cache,
                output,
                repeats[("copy", fraction)],
                source=packed,
                destination=staged,
                rows=len(case_set[fraction]["host_ids"]),
            )
    torch.cuda.synchronize(device)

    samples = {
        fraction: {"direct": [], "copy": [], "cpu_pack": []} for fraction in args.fractions
    }
    errors = {fraction: {} for fraction in args.fractions}
    check_rows = sorted({0, rows // 2, rows - 1} | set(range(0, rows, max(1, rows // 8))))
    rng = random.Random(args.seed + requests)
    eviction = torch.empty(64 * 2**20, dtype=torch.uint8, device=device)
    for iteration in range(args.rounds):
        set_id = iteration % args.index_sets
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            case = prepared[set_id][fraction]
            direct_indices.copy_(case["selected"].view(-1))
            copy_indices.copy_(case["remapped"])
            unique_rows = len(case["host_ids"])
            pack_start = time.perf_counter_ns()
            if unique_rows:
                torch.index_select(host, 0, case["host_ids"], out=packed[:unique_rows])
            samples[fraction]["cpu_pack"].append(
                (time.perf_counter_ns() - pack_start) / 1000
            )
            torch.cuda.synchronize(device)
            order = ["direct", "copy"]
            rng.shuffle(order)
            for mode in order:
                key = ("copy", fraction, set_id) if mode == "copy" else ("direct", fraction)
                eviction.fill_(iteration % 256)
                replay(graphs[key], repeats[(mode, fraction)])
                if iteration < args.index_sets:
                    expected = reference(cache, query, rope, case["selected"], check_rows)
                    actual = output[check_rows].float()
                    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
                    error = (actual - expected).abs().max().item()
                    errors[fraction][mode] = max(errors[fraction].get(mode, 0.0), error)
                samples[fraction][mode].append(
                    replay(graphs[key], repeats[(mode, fraction)])
                )
        if (iteration + 1) % 10 == 0 or iteration + 1 == args.rounds:
            print(f"requests={requests} rounds={iteration + 1}/{args.rounds}", flush=True)

    result = dict(requests=requests, mtp=args.mtp, query_rows=rows, cases=[])
    for fraction in args.fractions:
        count = round(fraction * topk)
        unique = [len(case_set[fraction]["host_ids"]) for case_set in prepared]
        case = dict(
            host_fraction=fraction,
            host_rows_per_query=count,
            host_refs=rows * count,
            host_ref_bytes=rows * count * 1152,
            copy_unique_rows=unique,
            copy_unique_bytes_median=sorted(unique)[len(unique) // 2] * 1152,
            repeats_per_graph={mode: repeats[(mode, fraction)] for mode in ("direct", "copy")},
            correctness_max_abs_error=errors[fraction],
            correctness_rows_checked=check_rows,
        )
        for name, values in samples[fraction].items():
            case[name] = summarize(values)
        result["cases"].append(case)
        print(
            f"requests={requests} rows={rows} host={fraction:.0%} "
            f"direct={case['direct']['median_us']:.2f}us copy={case['copy']['median_us']:.2f}us "
            f"cpu_pack={case['cpu_pack']['median_us']:.0f}us",
            flush=True,
        )
    del graphs, direct_wrapper, copy_wrapper, prepared
    del query, rope, output, direct_indices, copy_indices, workspace, copy_workspace, eviction
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--cpu-affinity", default="32-63,96-127")
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--mtp", type=int, default=8)
    parser.add_argument("--context", type=int, default=131072, help="KV tokens per request")
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--index-sets", type=int, default=4)
    parser.add_argument("--target-ms", type=float, default=10.0)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if 0 not in args.fractions:
        raise ValueError("A 0% all-HBM baseline is required")
    cpus = set()
    for span in args.cpu_affinity.split(","):
        limits = [int(value) for value in span.split("-")]
        cpus.update(range(limits[0], limits[-1] + 1))
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    max_requests = max(args.requests)
    tokens = max_requests * args.context
    allocation = MixedHBMHost(tokens, args.device)
    cache = allocation.tensor(args.device)
    # Fill both tiers on the GPU in chunks; the Host half is written over PCIe.
    for begin in range(0, tokens, 2**20):
        cache[begin : begin + 2**20].normal_()
    torch.cuda.synchronize(device)
    host = allocation.host_tensor()
    alias = StagedHBM(allocation, max_requests * args.mtp * round(max(args.fractions) * args.topk))
    alias_cache = alias.tensor(args.device)
    staged = alias_cache[tokens // 2 :]
    packed_rows = max_requests * args.mtp * round(max(args.fractions) * args.topk)
    packed = torch.empty((packed_rows, 1, 576), dtype=torch.bfloat16, pin_memory=True)

    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        device_name=torch.cuda.get_device_name(args.device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "temp/bench_glm52_host_batch.py",
                "include/flashinfer/attention/mla.cuh",
                "csrc/batch_mla_run.cu",
                "flashinfer/mla/_core.py",
            )
        },
        residency=allocation.residency,
        shape=dict(
            heads=HEADS,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            dtype="bfloat16",
            sm_scale=SM_SCALE,
            query_layout="each MTP token is an independent qo_len=1 row with its own top-k",
            kv_layout="request r owns HBM tokens [r*C/2,(r+1)*C/2) and Host tokens N/2+[r*C/2,(r+1)*C/2)",
        ),
        methods=dict(
            copy="CPU gathers unique Host rows of the whole batch into pinned memory; H2D DMA plus native attention in one CUDA graph",
            direct="Native attention directly reads mapped pinned Host DRAM through one VMM range",
        ),
        timing=dict(
            direct="GPU events around a CUDA graph of repeated attention calls",
            copy="GPU events around a CUDA graph; each call includes its H2D DMA",
            cpu_pack="CPU index_select into the pinned buffer, excluded from copy GPU time",
        ),
        exclusions=["top-k selection", "plan", "index remapping/update", "initialization", "correctness reference"],
        gpu_before=gpu_snapshot(),
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requests in args.requests:
        result["results"].append(
            benchmark_batch(args, requests, allocation, cache, host, alias_cache, staged, packed)
        )
        result["gpu_after"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    del cache, host, alias_cache, staged, packed
    gc.collect()
    torch.cuda.synchronize(device)
    alias.close()
    allocation.close()
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
