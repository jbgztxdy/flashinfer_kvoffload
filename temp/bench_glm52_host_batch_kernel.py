# Purpose: Compare Memcpy, native Direct, and the batched Host kernel for MTP sparse MLA decode,
#          with independent random top-k or top-k replayed from measured MTP overlap.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_batch_kernel.py \
#          --index-mode mtp --output temp/host_attention_batch_kernel/mtp.json
import argparse
import gc
import glob
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
from bench_glm52_host_batch import make_indices, make_wrapper, reference, replay  # noqa: E402
from bench_glm52_host_comparison import StagedHBM  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402
from bench_glm52_sparse_decode_p2p import gpu_snapshot, summarize  # noqa: E402

HEADS = 64
RECENT = 4096  # positions this close to the query token always stay in HBM (mtp mode)


def load_captures(pattern):
    captures = []
    for path in sorted(glob.glob(pattern)):
        data = torch.load(path)
        captures.append((Path(path).stem, data["selections"], data["group_starts"]))
    if not captures:
        raise FileNotFoundError(pattern)
    return captures


def make_mtp_indices(requests, mtp, context, topk, fractions, seed, captures, device, host_base):
    """Replay measured top-k of 8 consecutive tokens per request.

    Request r owns HBM pages r*C + p and Host pages host_base + r*C + p for positions p < C. Host
    residency is a property of the position, shared by all tokens of the request: positions older
    than RECENT are Host with probability theta, chosen so the request's mean Host share is the
    target fraction.
    """
    rng = random.Random(seed)
    combos = [(c, g) for c in range(len(captures)) for g in range(captures[c][1].shape[1])]
    rng.shuffle(combos)
    rows = []
    for r in range(requests):
        doc, group = combos[r % len(combos)]
        name, selections, starts = captures[doc]
        layer = rng.randrange(selections.shape[0])
        sel = selections[layer, group, :mtp].long().to(device)  # [mtp, topk] positions
        start = int(starts[group])
        assert sel.shape[1] == topk and int(sel.max()) < context
        rows.append((r, sel, start))
    result, actual = {}, {}
    for fraction in fractions:
        pages, host_refs = [], 0
        for r, sel, start in rows:
            far = sel < start - RECENT
            far_share = float(far.float().mean())
            theta = min(1.0, fraction / far_share) if far_share else 0.0
            # Same draw for every fraction, so residency grows monotonically with the fraction.
            u = torch.rand(context, generator=torch.Generator(device=device).manual_seed(seed * 7919 + r), device=device)
            host = far & (u[sel] < theta)
            host_refs += int(host.sum())
            pages.append(torch.where(host, host_base + r * context + sel, r * context + sel))
        result[fraction] = torch.cat(pages).int()
        actual[fraction] = host_refs / (requests * mtp * topk)
    return result, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--cpu-affinity", default="32-63,96-127")
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--mtp", type=int, default=8)
    parser.add_argument("--index-mode", choices=("random", "mtp"), default="mtp")
    parser.add_argument("--captures", default="temp/mtp_overlap/overlap_ctx65536_*.pt")
    parser.add_argument("--context", type=int, default=None, help="KV positions per request")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--modes", nargs="+", default=["direct", "copy", "batch"])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--index-sets", type=int, default=4)
    parser.add_argument("--target-ms", type=float, default=10.0)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--fetch-ctas", type=int, default=8)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--staging-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true", help="skip correctness (profiling builds)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if 0 not in args.fractions:
        raise ValueError("A 0% all-HBM baseline is required")
    if args.context is None:
        args.context = 65536 if args.index_mode == "mtp" else 57344
    cpus = set()
    for span in args.cpu_affinity.split(","):
        limits = [int(value) for value in span.split("-")]
        cpus.update(range(limits[0], limits[-1] + 1))
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    captures = load_captures(args.captures) if args.index_mode == "mtp" else None

    max_requests = max(args.requests)
    # random mode: each request owns C/2 HBM and C/2 Host pages; mtp mode: C of each.
    tokens = max_requests * args.context * (2 if args.index_mode == "mtp" else 1)
    allocation = MixedHBMHost(tokens, args.device)
    cache = allocation.tensor(args.device)
    for begin in range(0, tokens, 2**20):
        cache[begin : begin + 2**20].normal_()
    torch.cuda.synchronize(device)
    host = allocation.host_tensor()
    # Memcpy staging bound; mtp fractions are approximate, so leave headroom.
    max_host_rows = math.ceil(max_requests * args.mtp * args.topk * min(1.0, 1.5 * max(args.fractions)))
    alias = StagedHBM(allocation, max_host_rows)
    alias_cache = alias.tensor(args.device)
    staged = alias_cache[tokens // 2 :]
    packed = torch.empty((max_host_rows, 1, 576), dtype=torch.bfloat16, pin_memory=True)

    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        device_name=torch.cuda.get_device_name(args.device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "include", "csrc", "flashinfer"], text=True).strip()),
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "temp/bench_glm52_host_batch_kernel.py",
                "include/flashinfer/attention/mla_host_batch.cuh",
                "include/flashinfer/attention/mla.cuh",
                "csrc/batch_mla_run.cu",
                "flashinfer/mla/_core.py",
            )
        },
        residency=allocation.residency,
        methods=dict(
            copy="CPU gathers the batch's unique Host rows into pinned memory; H2D DMA plus native attention in one CUDA graph",
            direct="Native attention reads mapped Host DRAM for every reference",
            batch="Batched Host kernel: in-kernel dedup into HBM staging, fetch CTAs overlap PCIe with HBM attention",
            meta="Batched Host kernel reusing prebuilt metadata: no prepare, resolve or their grid barriers (experiment)",
            attn="Memcpy's attention with its H2D removed: the same rows read from HBM, no PCIe (timing only)",
        ),
        timing="GPU events around CUDA graphs of repeated calls; CPU gather for copy reported separately",
        gpu_before=gpu_snapshot(),
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requests in args.requests:
        result["results"].append(run_batch(args, requests, tokens, cache, host, alias_cache, staged, packed, captures, device))
        result["gpu_after"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    del cache, host, alias_cache, staged, packed
    gc.collect()
    torch.cuda.synchronize(device)
    alias.close()
    allocation.close()
    print(f"results={args.output}", flush=True)


def run_batch(args, requests, tokens, cache, host, alias_cache, staged, packed, captures, device):
    rows = requests * args.mtp
    topk = args.topk
    query = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((rows, HEADS, 64), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    indices = {mode: torch.empty(rows * topk, dtype=torch.int32, device=device) for mode in args.modes}
    wrappers = {
        mode: make_wrapper(rows, indices[mode], torch.empty(args.workspace_mib * 2**20, dtype=torch.uint8, device=device), topk)
        for mode in args.modes
    }
    for mode in ("batch", "meta"):
        if mode in wrappers:
            wrappers[mode].host_fetch_ctas = args.fetch_ctas
            wrappers[mode].host_staging_rows = args.staging_rows

    prepared, actual_fraction = [], {}
    for set_id in range(args.index_sets):
        seed = args.seed + 1000 * requests + set_id
        if args.index_mode == "mtp":
            selections, actual = make_mtp_indices(
                requests, args.mtp, args.context, topk, args.fractions, seed, captures, device, tokens // 2
            )
        else:
            selections = make_indices(requests, args.mtp, args.context, tokens, topk, args.fractions, seed, device)
            actual = {f: round(f * topk) / topk for f in args.fractions}
        case_set = {}
        for fraction, selected in selections.items():
            flat = selected.reshape(-1)
            mask = flat >= tokens // 2
            unique, inverse = torch.unique(flat[mask], return_inverse=True)
            remapped = flat.clone()
            remapped[mask] = (tokens // 2 + inverse).int()
            assert len(unique) <= len(packed), "Memcpy staging buffer too small"
            case_set[fraction] = dict(
                selected=flat, remapped=remapped, host_ids=(unique - tokens // 2).cpu(), host_refs=int(mask.sum())
            )
            actual_fraction.setdefault(fraction, []).append(actual[fraction])
        prepared.append(case_set)

    host_start = tokens // 2

    def call(mode, unique_rows=0):
        if mode in ("copy", "attn"):
            # attn: Memcpy's attention without its H2D, i.e. the same work with every selected row
            # already in HBM. Timing only: the staging rows hold whatever a previous call left.
            if unique_rows and mode == "copy":
                staged[:unique_rows].copy_(packed[:unique_rows], non_blocking=True)
            wrappers[mode].run(query, rope, alias_cache[..., :512], alias_cache[..., 512:], out=output)
        else:
            wrappers[mode].run(
                query, rope, cache[..., :512], cache[..., 512:], out=output,
                host_page_start=host_start if mode in ("batch", "meta") else None,
            )

    def prime_meta():
        # meta: the batched kernel reusing metadata a normal launch just built (see
        # kHostBatchReuseMeta), i.e. as if the top-k stage had emitted it.
        buffers = wrappers["meta"]._host_batch_buffers
        if buffers is not None:
            buffers[5][0] = 0
        call("meta")
        wrappers["meta"]._host_batch_buffers[5][0] = 1

    def load(case):
        for mode in args.modes:
            indices[mode].copy_(
                case["remapped"] if mode in ("copy", "attn") else case["selected"])
        if "meta" in args.modes:
            prime_meta()

    # Warm up (JIT, batch buffers) and size graphs to about target_ms each.
    first = prepared[0]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repeats = {}
    for fraction in args.fractions:
        load(first[fraction])
        for mode in args.modes:
            unique_rows = len(first[fraction]["host_ids"])
            for _ in range(2):
                call(mode, unique_rows)
            start.record()
            call(mode, unique_rows)
            end.record()
            end.synchronize()
            repeats[(mode, fraction)] = max(1, min(args.max_repeats, math.ceil(args.target_ms * 1000 / (start.elapsed_time(end) * 1000))))

    graphs = {}
    for fraction in args.fractions:
        for mode in args.modes:
            for set_id, case_set in enumerate(prepared if mode == "copy" else prepared[:1]):
                unique_rows = len(case_set[fraction]["host_ids"])

                def body(mode=mode, unique_rows=unique_rows):
                    call(mode, unique_rows)

                graph = torch.cuda.CUDAGraph()
                g_start = torch.cuda.Event(enable_timing=True, external=True)
                g_end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    g_start.record()
                    for _ in range(repeats[(mode, fraction)]):
                        body()
                    g_end.record()
                graphs[(mode, fraction, set_id if mode == "copy" else 0)] = (graph, g_start, g_end)
    torch.cuda.synchronize(device)

    samples = {f: {m: [] for m in args.modes + ["cpu_pack"]} for f in args.fractions}
    errors = {f: {} for f in args.fractions}
    check_rows = sorted({0, rows // 2, rows - 1} | set(range(0, rows, max(1, rows // 8))))
    rng = random.Random(args.seed + requests)
    eviction = torch.empty(64 * 2**20, dtype=torch.uint8, device=device)
    for iteration in range(args.rounds):
        set_id = iteration % args.index_sets
        fractions = args.fractions.copy()
        rng.shuffle(fractions)
        for fraction in fractions:
            case = prepared[set_id][fraction]
            load(case)
            unique_rows = len(case["host_ids"])
            pack_start = time.perf_counter_ns()
            if unique_rows:
                torch.index_select(host, 0, case["host_ids"], out=packed[:unique_rows])
            samples[fraction]["cpu_pack"].append((time.perf_counter_ns() - pack_start) / 1000)
            torch.cuda.synchronize(device)
            order = args.modes.copy()
            rng.shuffle(order)
            for mode in order:
                key = (mode, fraction, set_id if mode == "copy" else 0)
                eviction.fill_(iteration % 256)
                replay(graphs[key], repeats[(mode, fraction)])
                if iteration < args.index_sets and not args.no_check and mode != "attn":
                    expected = reference(cache, query, rope, case["selected"].view(rows, topk), check_rows)
                    actual = output[check_rows].float()
                    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
                    errors[fraction][mode] = max(errors[fraction].get(mode, 0.0), (actual - expected).abs().max().item())
                samples[fraction][mode].append(replay(graphs[key], repeats[(mode, fraction)]))
        if (iteration + 1) % 10 == 0 or iteration + 1 == args.rounds:
            print(f"requests={requests} rounds={iteration + 1}/{args.rounds}", flush=True)

    result = dict(requests=requests, mtp=args.mtp, query_rows=rows, cases=[])
    for fraction in args.fractions:
        refs = [cs[fraction]["host_refs"] for cs in prepared]
        uniq = [len(cs[fraction]["host_ids"]) for cs in prepared]
        case = dict(
            host_fraction=fraction,
            host_fraction_actual=sum(actual_fraction[fraction]) / len(actual_fraction[fraction]),
            host_refs=refs,
            host_unique=uniq,
            unique_ratio=sum(uniq) / max(1, sum(refs)),
            host_unique_bytes_median=sorted(uniq)[len(uniq) // 2] * 1152,
            host_ref_bytes_median=sorted(refs)[len(refs) // 2] * 1152,
            repeats_per_graph={m: repeats[(m, fraction)] for m in args.modes},
            correctness_max_abs_error=errors[fraction],
        )
        for name, values in samples[fraction].items():
            case[name] = summarize(values)
        result["cases"].append(case)
        print(
            f"requests={requests} rows={rows} host={fraction:.0%} (actual {case['host_fraction_actual']:.3f}, "
            f"unique {case['unique_ratio']:.3f}) "
            + " ".join(f"{m}={case[m]['median_us']:.1f}us" for m in args.modes),
            flush=True,
        )
    del graphs, wrappers, prepared, query, rope, output, indices, eviction
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
