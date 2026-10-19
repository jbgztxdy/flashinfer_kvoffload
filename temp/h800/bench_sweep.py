# Purpose: Batch sweep of sparse MLA decode with far KV pages on H800, per far tier
#          (host = pinned DRAM over PCIe, peer = another GPU's HBM over NVLink):
#          attn (all pages near: compute floor), copy (Memcpy: gather unique far rows, copy them
#          near, attend), direct (native kernel reading far pages in place), batch (our kernel).
#          Top-k replays measured DSA-proxy MTP captures. One decode step per case, no cache.
# Usage: source temp/h800/env_job.sh; python temp/h800/bench_sweep.py --tier host --mtp 8 \
#          --requests 1 2 4 --output temp/h800/results/x.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiers  # noqa: E402
from bench_glm52_host_batch import make_wrapper, reference, replay  # noqa: E402
from bench_glm52_sparse_decode_p2p import gpu_snapshot, summarize  # noqa: E402

HEADS = 64


def probe_bandwidth(kv, main, alias, device, gib=1.0):
    """Far-tier read bandwidth: one contiguous DMA copy and an SM gather of random rows."""
    rows = min(int(gib * 2**30) // tiers.ROW_BYTES, kv.far, kv.staging)
    far = main[kv.near : kv.near + rows]
    dst = alias[kv.near : kv.near + rows]
    ids = torch.randperm(kv.far, device=device)[:rows]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = {}
    for name, fn in (("dma_contiguous", lambda: dst.copy_(far)),
                     ("sm_gather_random", lambda: torch.index_select(main[kv.near :], 0, ids, out=dst))):
        fn()
        times = []
        for _ in range(5):
            start.record(); fn(); end.record(); end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        out[name + "_GBps"] = rows * tiers.ROW_BYTES / sorted(times)[2] / 1e9
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("host", "peer"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peer", type=int, default=1)
    parser.add_argument("--requests", nargs="+", type=int, required=True)
    parser.add_argument("--mtp", type=int, default=8)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--modes", nargs="+", default=["attn", "copy", "direct", "batch"])
    parser.add_argument("--captures", default="temp/mtp_overlap/overlap_ctx65536_*.pt")
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--near-requests", type=int, default=128, help="requests with distinct near pages")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--index-sets", type=int, default=2)
    parser.add_argument("--target-ms", type=float, default=8.0)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--fetch-ctas", type=int, default=8)
    parser.add_argument("--host-item-entries", type=int, default=64)
    parser.add_argument("--no-host-split", action="store_true", help="keep Host entries in row chunks")
    parser.add_argument("--no-stage", action="store_true", help="read far pages in place (no staging)")
    parser.add_argument("--staging-rows", type=int, default=None, help="batch staging rows (None: default)")
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if 0 not in args.fractions:
        raise ValueError("a 0% all-near case is required")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    captures = tiers.load_captures(args.captures)
    max_requests = max(args.requests)
    near_requests = min(args.near_requests, max_requests)
    max_far_rows = math.ceil(max_requests * args.mtp * args.topk * min(1.0, 1.6 * max(args.fractions)))
    kv = tiers.TieredKV(near_requests * args.context, max_requests * args.context, max_far_rows,
                        args.device, args.tier, args.peer)
    main_kv, alias = kv.main(), kv.alias()
    for begin in range(0, kv.near + kv.far, 2**20):
        main_kv[begin : begin + 2**20].normal_()
    torch.cuda.synchronize(device)
    far_cpu = kv.far_cpu() if args.tier == "host" else None
    packed = torch.empty((max_far_rows, 1, 576), dtype=torch.bfloat16, pin_memory=True) if far_cpu is not None else None
    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        device_name=torch.cuda.get_device_name(args.device),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "include", "csrc", "flashinfer"], text=True).strip()),
        source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in (
            "temp/h800/bench_sweep.py", "temp/h800/tiers.py", "include/flashinfer/attention/mla_host_batch.cuh")},
        residency=kv.residency,
        bandwidth=probe_bandwidth(kv, main_kv, alias, device),
        methods=dict(
            attn="native FA2 MLA attention, every selected row near (compute floor, no transfer)",
            copy=("Memcpy: unique far rows gathered (host: CPU index_select into pinned memory, not timed; "
                  "peer: GPU index_select, timed) then H2D/D2D-copied near, then native attention"),
            direct="native attention reading far rows in place, every reference separately",
            batch="our batched Host kernel: in-kernel dedup, fetch CTAs overlap far reads with near attention",
        ),
        gpu_before=gpu_snapshot(), results=[],
    )
    print("bandwidth", result["bandwidth"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requests in args.requests:
        result["results"].append(run_batch(args, requests, kv, main_kv, alias, far_cpu, packed, captures, device))
        result["gpu_after"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    del main_kv, alias, far_cpu, packed
    gc.collect()
    torch.cuda.synchronize(device)
    kv.close()
    print(f"results={args.output}", flush=True)


def run_batch(args, requests, kv, main_kv, alias, far_cpu, packed, captures, device):
    rows, topk = requests * args.mtp, args.topk
    query = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((rows, HEADS, 64), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    indices = {m: torch.empty(rows * topk, dtype=torch.int32, device=device) for m in args.modes}
    wrappers = {m: make_wrapper(rows, indices[m], torch.empty(args.workspace_mib * 2**20, dtype=torch.uint8, device=device), topk)
                for m in args.modes}
    if "batch" in wrappers:
        wrappers["batch"].host_fetch_ctas = args.fetch_ctas
        wrappers["batch"].host_staging_rows = args.staging_rows
        wrappers["batch"].host_item_entries = args.host_item_entries
        wrappers["batch"].host_split = not args.no_host_split
        wrappers["batch"].host_stage = not args.no_stage
    staged = alias[kv.near :]
    far_view = main_kv[kv.near :]
    near_requests = min(args.near_requests, requests)

    prepared = []
    for set_id in range(args.index_sets):
        case_set = {}
        for fraction in args.fractions:
            reqs = tiers.Requests(captures, requests, args.context, kv.near, near_requests, fraction,
                                  args.seed + 1000 * requests + set_id, "cpu")
            pages, _, far_refs = reqs.rows(0, args.mtp)
            flat = pages.view(-1)
            mask = flat >= kv.near
            unique, inverse = torch.unique(flat[mask], return_inverse=True)
            remapped = flat.clone()
            remapped[mask] = (kv.near + inverse).int()
            assert len(unique) <= kv.staging
            far_ids = (unique - kv.near)
            case_set[fraction] = dict(selected=flat.to(device), remapped=remapped.to(device), far_ids=far_ids,
                                      far_ids_dev=far_ids.to(device), far_refs=far_refs,
                                      far_fraction=far_refs / (rows * topk))
        prepared.append(case_set)

    def call(mode, case):
        n = len(case["far_ids"])
        if mode in ("copy", "attn"):
            if mode == "copy" and n:
                if far_cpu is not None:
                    staged[:n].copy_(packed[:n], non_blocking=True)
                else:
                    torch.index_select(far_view, 0, case["far_ids_dev"], out=staged[:n])
            wrappers[mode].run(query, rope, alias[..., :512], alias[..., 512:], out=output)
        else:
            wrappers[mode].run(query, rope, main_kv[..., :512], main_kv[..., 512:], out=output,
                               host_page_start=kv.near if mode == "batch" else None)

    def load(case):
        for mode in args.modes:
            indices[mode].copy_(case["remapped"] if mode in ("copy", "attn") else case["selected"])

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repeats, graphs = {}, {}
    for fraction in args.fractions:
        case = prepared[0][fraction]
        load(case)
        for mode in args.modes:
            for _ in range(2):
                call(mode, case)
            start.record(); call(mode, case); end.record(); end.synchronize()
            repeats[(mode, fraction)] = max(1, min(args.max_repeats, math.ceil(args.target_ms / start.elapsed_time(end))))
    for fraction in args.fractions:
        for mode in args.modes:
            for set_id, case_set in enumerate(prepared):
                case = case_set[fraction]
                graph = torch.cuda.CUDAGraph()
                g_start = torch.cuda.Event(enable_timing=True, external=True)
                g_end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    g_start.record()
                    for _ in range(repeats[(mode, fraction)]):
                        call(mode, case)
                    g_end.record()
                graphs[(mode, fraction, set_id)] = (graph, g_start, g_end)
    torch.cuda.synchronize(device)

    samples = {f: {m: [] for m in args.modes + ["cpu_gather"]} for f in args.fractions}
    errors = {f: {} for f in args.fractions}
    check_rows = sorted({0, rows // 2, rows - 1} | set(range(0, rows, max(1, rows // 6))))
    rng = random.Random(args.seed + requests)
    eviction = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)
    for iteration in range(args.rounds):
        set_id = iteration % args.index_sets
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            case = prepared[set_id][fraction]
            load(case)
            n = len(case["far_ids"])
            if far_cpu is not None and n:
                t0 = time.perf_counter_ns()
                torch.index_select(far_cpu, 0, case["far_ids"], out=packed[:n])
                samples[fraction]["cpu_gather"].append((time.perf_counter_ns() - t0) / 1000)
            torch.cuda.synchronize(device)
            order = args.modes.copy()
            rng.shuffle(order)
            for mode in order:
                key = (mode, fraction, set_id)
                eviction.fill_(iteration % 256)
                replay(graphs[key], repeats[(mode, fraction)])
                # attn reads whatever the staging rows hold: timing only.
                if iteration < args.index_sets and not args.no_check and mode != "attn":
                    expected = reference(main_kv, query, rope, case["selected"].view(rows, topk), check_rows)
                    actual = output[check_rows].float()
                    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
                    errors[fraction][mode] = max(errors[fraction].get(mode, 0.0), (actual - expected).abs().max().item())
                samples[fraction][mode].append(replay(graphs[key], repeats[(mode, fraction)]))

    result = dict(requests=requests, mtp=args.mtp, query_rows=rows, cases=[])
    for fraction in args.fractions:
        refs = [cs[fraction]["far_refs"] for cs in prepared]
        uniq = [len(cs[fraction]["far_ids"]) for cs in prepared]
        case = dict(
            far_fraction=fraction,
            far_fraction_actual=sum(cs[fraction]["far_fraction"] for cs in prepared) / len(prepared),
            far_refs=refs, far_unique=uniq, unique_ratio=sum(uniq) / max(1, sum(refs)),
            far_unique_bytes=[u * tiers.ROW_BYTES for u in uniq],
            repeats_per_graph={m: repeats[(m, fraction)] for m in args.modes},
            correctness_max_abs_error=errors[fraction],
        )
        for name, values in samples[fraction].items():
            if values:
                case[name] = summarize(values)
        result["cases"].append(case)
        print(f"tier={args.tier} requests={requests} rows={rows} far={fraction:.0%} "
              f"(actual {case['far_fraction_actual']:.3f}, unique {case['unique_ratio']:.3f}) "
              + " ".join(f"{m}={case[m]['median_us']:.1f}us" for m in args.modes), flush=True)
    del graphs, wrappers, prepared, query, rope, output, indices, eviction
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
