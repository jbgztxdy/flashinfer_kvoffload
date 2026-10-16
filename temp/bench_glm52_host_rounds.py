# Purpose: Measure what a cross-decode-round persistent HBM cache would buy, for all three paths
#          (Memcpy / Direct / batched Host kernel), on real consecutive-round DSA-like top-k.
#
# Method: the cache is emulated by page placement, not implemented. Each request owns an HBM copy
# and a Host copy of every position (see make_mtp_indices). A token the cache already holds is
# addressed through its HBM copy, a miss through its Host copy; both halves are filled with the
# same bytes, so outputs are identical either way and only PCIe traffic differs. This measures the
# steady-state transfer benefit exactly, but NOT the cost of hit detection / eviction, and it
# models an unbounded cache (see --capacity for a bounded LRU variant).
#
# Usage: PYTHONPATH=. python temp/bench_glm52_host_rounds.py --output temp/host_attention_rounds/r.json
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
from bench_glm52_host_batch import make_wrapper, reference, replay  # noqa: E402
from bench_glm52_host_batch_kernel import load_captures  # noqa: E402
from bench_glm52_host_comparison import StagedHBM  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402
from bench_glm52_sparse_decode_p2p import gpu_snapshot, summarize  # noqa: E402

HEADS, MTP, RECENT = 64, 8, 4096


def build_rounds(args, captures, device, tokens):
    """Per round: the selected pages, with cached ones addressed in HBM.

    Host residency is a property of the position and fixed across rounds (a token does not move
    tiers mid-request); the cache only changes which copy we address.
    """
    rng = random.Random(args.seed)
    combos = [(c, l, g) for c in range(len(captures))
              for l in range(captures[c][1].shape[0]) for g in range(captures[c][1].shape[1])]
    rng.shuffle(combos)
    rounds = args.rounds_per_request
    per_round = []            # [round][request] -> LongTensor[MTP, topk] of absolute pages
    seen = [set() for _ in range(args.requests)]
    stats = []
    for t in range(rounds):
        pages, miss_frac = [], []
        for r in range(args.requests):
            doc, layer, group = combos[r % len(combos)]
            name, selections, starts = captures[doc]
            sel = selections[layer, group, t * MTP : (t + 1) * MTP].long()
            start = int(starts[group]) + t * MTP
            # Host residency: far positions, fixed per (request, position) across rounds.
            gen = torch.Generator().manual_seed(args.seed * 7919 + r)
            u = torch.rand(args.context, generator=gen)
            far_share = float((sel < start - RECENT).float().mean())
            theta = min(1.0, args.fraction / far_share) if far_share else 0.0
            host_pos = (sel < start - RECENT) & (u[sel] < theta)
            # Cache hit -> address the HBM copy of the same position.
            flat = sel.reshape(-1)
            hit = torch.tensor([int(p) in seen[r] for p in flat.tolist()]).view_as(sel)
            use_host = host_pos & ~(hit if args.cache else torch.zeros_like(hit))
            base = r * args.context
            pages.append(torch.where(use_host, tokens // 2 + base + sel, base + sel))
            miss_frac.append(float(use_host.float().mean()))
            if args.cache:
                seen[r].update(flat[host_pos.reshape(-1)].tolist())
        per_round.append(torch.cat(pages).reshape(-1).int().to(device))
        stats.append(sum(miss_frac) / len(miss_frac))
    return per_round, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--cpu-affinity", default="32-63,96-127")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--rounds-per-request", type=int, default=4)
    parser.add_argument("--captures", default="temp/mtp_overlap/rounds_ctx65536_*.pt")
    parser.add_argument("--modes", nargs="+", default=["direct", "copy", "batch"])
    parser.add_argument("--reps", type=int, default=20, help="timing repeats per round")
    parser.add_argument("--fetch-ctas", type=int, default=8)
    parser.add_argument("--workspace-mib", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cpus = set()
    for span in args.cpu_affinity.split(","):
        limits = [int(v) for v in span.split("-")]
        cpus.update(range(limits[0], limits[-1] + 1))
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    captures = load_captures(args.captures)

    rows = args.requests * MTP
    tokens = 2 * args.requests * args.context
    allocation = MixedHBMHost(tokens, args.device)
    cache = allocation.tensor(args.device)
    for b in range(0, tokens, 2**20):
        cache[b : b + 2**20].normal_()
    # The HBM copy and the Host copy of a position must hold the same bytes, so that addressing
    # either one gives the same attention output.
    half = tokens // 2
    for b in range(0, half, 2**20):
        n = min(2**20, half - b)
        cache[b : b + n].copy_(cache[half + b : half + b + n])
    torch.cuda.synchronize(device)
    host = allocation.host_tensor()
    max_host = rows * args.topk
    alias = StagedHBM(allocation, max_host)
    alias_cache = alias.tensor(args.device)
    staged = alias_cache[half:]
    packed = torch.empty((max_host, 1, 576), dtype=torch.bfloat16, pin_memory=True)

    q = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((rows, HEADS, 64), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q)
    idx = {m: torch.empty(rows * args.topk, dtype=torch.int32, device=device) for m in args.modes}
    wrappers = {
        m: make_wrapper(rows, idx[m], torch.empty(args.workspace_mib * 2**20, dtype=torch.uint8,
                                                  device=device), args.topk)
        for m in args.modes
    }
    if "batch" in wrappers:
        wrappers["batch"].host_fetch_ctas = args.fetch_ctas

    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        device_name=torch.cuda.get_device_name(args.device),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(
            ["git", "status", "--porcelain", "include", "csrc", "flashinfer"], text=True).strip()),
        source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in (
            "temp/bench_glm52_host_rounds.py", "include/flashinfer/attention/mla_host_batch.cuh")},
        method="Cross-round cache emulated by page placement (HBM copy = hit, Host copy = miss); "
               "measures transfer benefit only, excludes hit-detection and eviction cost; "
               "cache is unbounded over the measured rounds",
        variants=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for use_cache in (False, True):
        args.cache = use_cache
        per_round, miss = build_rounds(args, captures, device, tokens)
        samples = {t: {m: [] for m in args.modes} for t in range(args.rounds_per_request)}
        errors = {}
        check = sorted({0, rows // 2, rows - 1} | set(range(0, rows, max(1, rows // 6))))
        for t, sel in enumerate(per_round):
            flat = sel
            mask = flat >= half
            uniq, inv = torch.unique(flat[mask], return_inverse=True)
            remap = flat.clone()
            remap[mask] = (half + inv).int()
            n_uniq = len(uniq)
            if n_uniq:
                torch.index_select(host, 0, (uniq - half).cpu(), out=packed[:n_uniq])
            for m in args.modes:
                idx[m].copy_(remap if m == "copy" else flat)
            torch.cuda.synchronize(device)

            def call(m):
                if m == "copy":
                    if n_uniq:
                        staged[:n_uniq].copy_(packed[:n_uniq], non_blocking=True)
                    wrappers[m].run(q, rope, alias_cache[..., :512], alias_cache[..., 512:], out=out)
                else:
                    wrappers[m].run(q, rope, cache[..., :512], cache[..., 512:], out=out,
                                    host_page_start=half if m == "batch" else None)

            for m in args.modes:
                for _ in range(2):
                    call(m)
                torch.cuda.synchronize(device)
                expected = reference(cache, q, rope, flat.view(rows, args.topk), check)
                torch.testing.assert_close(out[check].float(), expected, rtol=1e-2, atol=1e-2)
                errors[f"r{t}_{m}"] = (out[check].float() - expected).abs().max().item()
                graph = torch.cuda.CUDAGraph()
                s = torch.cuda.Event(enable_timing=True, external=True)
                e = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    s.record()
                    for _ in range(args.reps):
                        call(m)
                    e.record()
                for _ in range(5):
                    samples[t][m].append(replay((graph, s, e), args.reps))
                del graph
            print(f"{'cache' if use_cache else 'nocache'} round {t}: miss {miss[t]:.1%}  "
                  + "  ".join(f"{m}={sorted(samples[t][m])[2]:.0f}us" for m in args.modes),
                  flush=True)
        result["variants"].append(dict(
            cache=use_cache,
            rounds=[dict(round=t, host_miss_fraction=miss[t],
                         **{m: summarize(samples[t][m]) for m in args.modes}) for t in samples],
            correctness_max_abs_error=max(errors.values()),
        ))
        result["gpu"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    del wrappers, cache, alias_cache, staged, packed, q, rope, out, idx
    gc.collect()
    torch.cuda.synchronize(device)
    alias.close()
    allocation.close()
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
