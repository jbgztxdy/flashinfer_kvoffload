# Purpose: Consecutive decode steps of one sparse MLA layer with a cross-step HBM cache, on H800:
#          our batched kernel with its in-kernel LRU (staged or in place) against SGLang HiSparse
#          (its LRU swap-in kernel, then FlashMLA sparse attention), both over the same far KV
#          (pinned Host DRAM over PCIe, or a peer GPU's HBM over NVLink) and the same top-k.
#          Top-k replays measured consecutive-step DSA-proxy captures (rounds_ctx65536_*).
#          Each pass starts cold and runs every step once; per-step times are medians over passes.
# Usage: source temp/h800/env_job.sh; python temp/h800/bench_rounds.py --tier host --mtp 1 \
#          --requests 1 16 --output temp/h800/results/rounds.json
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiers  # noqa: E402
from bench_glm52_host_batch import make_wrapper, reference  # noqa: E402
from bench_glm52_sparse_decode_p2p import gpu_snapshot, summarize  # noqa: E402
from bench_sweep import probe_bandwidth  # noqa: E402

HEADS, SM_SCALE = 64, 1 / 16


class HiSparse:
    """Single-layer HiSparse: per-request LRU device buffer of `hot` slots plus the newest-token
    slot, filled by SGLang's load_cache_to_device_buffer_mla from the far pages, then FlashMLA."""

    def __init__(self, requests, hot, topk, far_view, context, device, block_size):
        from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd
        self.swap_in, self.attention = load_cache_to_device_buffer_mla, flash_mla_sparse_fwd
        self.requests, self.hot, self.topk, self.block_size = requests, hot, topk, block_size
        padded = hot + 1
        i32 = dict(dtype=torch.int32, device=device)
        self.buffer = torch.zeros((requests * padded, 576), dtype=torch.bfloat16, device=device)
        self.buffer_locs = (torch.arange(requests, **i32).view(-1, 1) * padded + torch.arange(padded, **i32)).contiguous()
        self.buffer_tokens = torch.full((requests, padded), -1, **i32)
        self.lru_init = torch.arange(hot, dtype=torch.int16, device=device)
        self.lru = self.lru_init.repeat(requests, 1).contiguous()
        self.host_locs = (torch.arange(requests, dtype=torch.int64, device=device).view(-1, 1) * context
                          + torch.arange(context + 1, dtype=torch.int64, device=device)).contiguous()
        self.host_cache = far_view.view(-1, 576)
        self.req_pool = torch.arange(requests, dtype=torch.int64, device=device)
        self.top_k_tokens = torch.empty((requests, topk), **i32)
        self.locs = torch.empty((requests, topk), **i32)
        self.seq_lens = torch.empty(requests, dtype=torch.int64, device=device)
        self.num_real = torch.tensor([requests], **i32)

    def reset(self):
        self.buffer_tokens.fill_(-1)
        self.lru.copy_(self.lru_init.expand_as(self.lru))

    def step(self, q):
        self.swap_in(
            top_k_tokens=self.top_k_tokens, device_buffer_tokens=self.buffer_tokens,
            host_cache_locs=self.host_locs, device_buffer_locs=self.buffer_locs,
            host_cache=self.host_cache, device_buffer=self.buffer, top_k_device_locs=self.locs,
            req_pool_indices=self.req_pool, seq_lens=self.seq_lens, lru_slots=self.lru,
            item_size_bytes=1152, num_top_k=self.topk, hot_buffer_size=self.hot, page_size=1,
            block_size=self.block_size, num_real_reqs=self.num_real)
        return self.attention(q, self.buffer.view(-1, 1, 576), self.locs.view(-1, 1, self.topk), SM_SCALE, 512)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("host", "peer"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peer", type=int, default=1)
    parser.add_argument("--requests", nargs="+", type=int, required=True)
    parser.add_argument("--mtp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="decode steps (default: all captured)")
    parser.add_argument("--fraction", type=float, default=1.0, help="far share; 1 = all far but the newest token")
    parser.add_argument("--capacity", type=int, default=4096, help="our LRU slots per request = HiSparse hot buffer")
    parser.add_argument("--modes", nargs="+",
                        default=["ours_stage", "ours_inplace", "ours_nocache", "hisparse", "floor_fa2", "floor_flashmla"])
    parser.add_argument("--captures", default="temp/mtp_overlap/rounds_ctx65536_*.pt")
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--near-requests", type=int, default=128)
    parser.add_argument("--passes", type=int, default=7)
    parser.add_argument("--fetch-ctas", type=int, default=8)
    parser.add_argument("--hisparse-block", type=int, default=960, help="SGLang default swap_in_block_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "hisparse" in args.modes and args.fraction < 1.0:
        # HiSparse keeps everything but the newest token on the far tier.
        raise ValueError("hisparse needs --fraction 1")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    captures = tiers.load_captures(args.captures)
    tokens = captures[0][1].shape[2]
    steps = args.steps or tokens // args.mtp
    assert steps * args.mtp <= tokens
    max_requests = max(args.requests)
    near_requests = min(args.near_requests, max_requests)
    rows_max = max_requests * args.mtp
    kv = tiers.TieredKV(near_requests * args.context, max_requests * args.context, rows_max * args.topk,
                        args.device, args.tier, args.peer)
    main_kv, alias = kv.main(), kv.alias()
    for begin in range(0, kv.near + kv.far, 2**20):
        main_kv[begin : begin + 2**20].normal_()
    torch.cuda.synchronize(device)
    result = dict(
        configuration=vars(args) | {"output": str(args.output), "steps": steps},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        device_name=torch.cuda.get_device_name(args.device),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "include", "csrc", "flashinfer"], text=True).strip()),
        source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in (
            "temp/h800/bench_rounds.py", "temp/h800/tiers.py", "include/flashinfer/attention/mla_host_batch.cuh")},
        residency=kv.residency, bandwidth=probe_bandwidth(kv, main_kv, alias, device),
        methods=dict(
            ours_stage="our kernel, LRU pool of `capacity` per request, misses staged (PCIe layout)",
            ours_inplace="our kernel, LRU pool of `capacity` per request, misses read in place, fetch CTAs fill the cache",
            ours_nocache="our kernel without cache (staged on host tier, in place on peer tier)",
            hisparse=("SGLang HiSparse: load_cache_to_device_buffer_mla (per-request LRU of `capacity` + newest "
                      "slot, one block per request) then sgl_kernel FlashMLA sparse fwd; MTP rows run one "
                      "draft token after another, since the HiSparse decode path takes one row per request"),
            floor_fa2="native FA2 MLA with every selected row near: compute floor of our kernel",
            floor_flashmla="FlashMLA sparse fwd with every selected row in HBM: compute floor of HiSparse",
            direct_lru=("Direct with the same LRU: native attention with hits near and misses read in place, then a "
                        "GPU gather of the misses into the cache for the next step"),
            memcpy_lru=("Memcpy with the same per-request step LRU: copy only each step's missed pages near "
                        "(host: untimed CPU gather + H2D DMA; peer: GPU gather), then native attention"),
        ),
        gpu_before=gpu_snapshot(), results=[],
    )
    print("bandwidth", result["bandwidth"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requests in args.requests:
        result["results"].append(run_requests(args, requests, steps, kv, main_kv, alias, captures, device))
        result["gpu_after"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    del main_kv, alias
    gc.collect()
    torch.cuda.synchronize(device)
    kv.close()
    print(f"results={args.output}", flush=True)


def run_requests(args, requests, steps, kv, main_kv, alias, captures, device):
    mtp, topk = args.mtp, args.topk
    rows = requests * mtp
    near_requests = min(args.near_requests, requests)
    reqs = tiers.Requests(captures, requests, args.context, kv.near, near_requests, args.fraction,
                          args.seed + requests, "cpu")
    plan = []
    for t in range(steps):
        pages, positions, far_refs = reqs.rows(t * mtp, mtp)
        flat = pages.view(-1)
        mask = flat >= kv.near
        unique, inverse = torch.unique(flat[mask], return_inverse=True)
        remapped = flat.clone()
        remapped[mask] = (kv.near + inverse).int()
        # HiSparse: per draft token, positions [requests, topk] and the newest token's page.
        newest = torch.tensor([s + t * mtp for s in reqs.start]).view(-1, 1) + torch.arange(mtp)
        # The newest token is always near (see tiers.Requests.rows).
        newest_page = (torch.arange(requests).view(-1, 1) % near_requests) * args.context + newest
        far_by_request = [set(x for x in row.tolist() if x >= kv.near) for row in pages.view(requests, -1)]
        plan.append(dict(selected=flat.to(device), remapped=remapped.to(device), far_refs=far_refs,
                         far_by_request=far_by_request,
                         far_unique_step=len(unique), positions=positions.view(requests, mtp, topk).to(device),
                         newest=newest.to(device), newest_page=newest_page.to(device),
                         seq_lens=(newest + 1).to(device)))
    if "memcpy_lru" in args.modes or "direct_lru" in args.modes:
        # Memcpy with the same per-request step LRU (as the kernel): each step copies only the
        # missed pages near (host: CPU gather into pinned memory, untimed, then one H2D DMA;
        # peer: a GPU gather), then the native kernel attends with every row near.
        stamps = [dict() for _ in range(requests)]
        for t, p in enumerate(plan):
            missed = []
            for r, need in enumerate(p["far_by_request"]):
                st = stamps[r]
                miss = [x for x in need if x not in st]
                for x in need:
                    if x in st:
                        st[x] = t
                free = args.capacity - len(st)
                evictable = sorted((v, x) for x, v in st.items() if v != t)
                for i, x in enumerate(miss):
                    if i < free + len(evictable):
                        if i >= free:
                            del st[evictable[i - free][1]]
                        st[x] = t
                missed += miss
            ids = torch.tensor(sorted(missed), dtype=torch.int64) - kv.near
            p["miss_ids"] = ids
            p["miss_ids_dev"] = ids.to(device)
            # direct_lru: hits read near (any near page: timing only), misses in place.
            flat = p["selected"].long()
            far = flat >= kv.near
            is_miss = torch.zeros(kv.far, dtype=torch.bool, device=device)
            is_miss[p["miss_ids_dev"]] = True
            hit = far & ~is_miss[(flat - kv.near).clamp(min=0)]
            p["direct_lru_idx"] = torch.where(hit, flat % kv.near, flat).int()
        far_view = main_kv[kv.near :]
        staged = alias[kv.near :]
        far_cpu = kv.far_cpu() if args.tier == "host" else None
        packed = (torch.empty((max(len(p["miss_ids"]) for p in plan), 1, 576), dtype=torch.bfloat16, pin_memory=True)
                  if far_cpu is not None else None)
    query = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16, device=device)
    rope = torch.randn((rows, HEADS, 64), dtype=torch.bfloat16, device=device)
    q_cat = torch.cat([query, rope], -1).view(requests, mtp, HEADS, 576)
    output = torch.empty_like(query)
    check_rows = sorted({0, rows // 2, rows - 1})
    pools = torch.arange(requests + 1, dtype=torch.int32) * mtp
    hisparse = None
    ours = {}
    for mode in args.modes:
        if mode.startswith("ours") or mode in ("floor_fa2", "memcpy_lru", "direct_lru"):
            indices = torch.empty(rows * topk, dtype=torch.int32, device=device)
            # Plans keep their metadata in the workspace, so each wrapper needs its own.
            w = make_wrapper(rows, indices, torch.empty(256 * 2**20, dtype=torch.uint8, device=device), topk)
            w.host_fetch_ctas = args.fetch_ctas
            w.host_stage = mode == "ours_stage" or (mode == "ours_nocache" and args.tier == "host")
            if mode in ("ours_stage", "ours_inplace"):
                w.set_host_cache(pools, args.capacity)
            ours[mode] = (w, indices)
        elif mode == "hisparse":
            hisparse = HiSparse(requests, args.capacity, topk, main_kv[kv.near :], args.context, device, args.hisparse_block)
    compact = alias  # floor_flashmla reads the selected rows compacted into the staging rows

    def load(mode, step):
        p = plan[step]
        if mode in ours:
            ours[mode][1].copy_(p["remapped"] if mode in ("floor_fa2", "memcpy_lru")
                                else p["direct_lru_idx"] if mode == "direct_lru" else p["selected"])
        if mode == "memcpy_lru" and far_cpu is not None and len(p["miss_ids"]):
            torch.index_select(far_cpu, 0, p["miss_ids"], out=packed[: len(p["miss_ids"])])

    def run(mode, step, j=None):
        p = plan[step]
        if mode in ("floor_fa2", "memcpy_lru"):
            n = len(p["miss_ids"]) if mode == "memcpy_lru" else 0
            if n and far_cpu is not None:
                staged[:n].copy_(packed[:n], non_blocking=True)
            elif n:
                torch.index_select(far_view, 0, p["miss_ids_dev"], out=staged[:n])
            ours[mode][0].run(query, rope, alias[..., :512], alias[..., 512:], out=output)
        elif mode == "direct_lru":
            ours[mode][0].run(query, rope, main_kv[..., :512], main_kv[..., 512:], out=output)
            if len(p["miss_ids"]):
                torch.index_select(far_view, 0, p["miss_ids_dev"], out=staged[: len(p["miss_ids"])])
        elif mode in ours:
            ours[mode][0].run(query, rope, main_kv[..., :512], main_kv[..., 512:], out=output, host_page_start=kv.near)
        elif mode == "floor_flashmla":
            # Every selected row in HBM: positions index the compacted rows directly.
            for jj in range(mtp):
                torch.ops.sgl_kernel.sparse_prefill_fwd.default(
                    q_cat[:, jj].contiguous(), compact.view(-1, 1, 576),
                    (p["remapped"].view(requests, mtp, topk)[:, jj]).contiguous().view(requests, 1, topk),
                    SM_SCALE, 512, None, None)

    def hisparse_step(step, check=False):
        p = plan[step]
        outs = []
        for j in range(mtp):
            hisparse.top_k_tokens.copy_(p["positions"][:, j])
            hisparse.seq_lens.copy_(p["seq_lens"][:, j])
            # The newest token's KV is in its reserved slot (written by the decode step itself).
            hisparse.buffer.view(requests, -1, 576)[:, args.capacity].copy_(main_kv[p["newest_page"][:, j].long(), 0])
            outs.append(hisparse.step(q_cat[:, j].contiguous()))
        return torch.stack(outs, 1).view(rows, HEADS, 512) if check else None

    def reset(mode):
        if mode in ("ours_stage", "ours_inplace"):
            ours[mode][0].release_host_cache(torch.arange(requests))
        elif mode == "hisparse":
            hisparse.reset()

    # Pass 0 checks every step against the reference; it also builds the buffers.
    # memcpy_lru attends over compacted rows that hold only this step's misses: timing only.
    errors = {m: 0.0 for m in args.modes if not m.startswith("floor") and not m.endswith("_lru")}
    for mode in args.modes:
        reset(mode)
        for step in range(steps):
            p = plan[step]
            expected = reference(main_kv, query, rope, p["selected"].view(rows, topk), check_rows)
            if mode == "hisparse":
                out = hisparse_step(step, check=True)
            else:
                load(mode, step)
                run(mode, step)
                out = output
            if mode in errors:
                actual = out[check_rows].float()
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2,
                                           msg=lambda m, mode=mode, step=step: f"{mode} step {step}: {m}")
                errors[mode] = max(errors[mode], (actual - expected).abs().max().item())
    torch.cuda.synchronize(device)

    samples = {m: [[] for _ in range(steps)] for m in args.modes}
    misses = {m: [[] for _ in range(steps)] for m in ("ours_stage", "ours_inplace") if m in args.modes}
    sleep_cycles = int(2e9 * (2e-3 + steps * 3e-4 * (mtp if "hisparse" in args.modes else 1)))
    for _ in range(args.passes):
        for mode in args.modes:
            reset(mode)
            torch.cuda.synchronize(device)
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
            uniq = torch.zeros(steps, dtype=torch.int32, device=device)
            torch.cuda._sleep(sleep_cycles)  # keeps the GPU busy while the steps are enqueued
            for step in range(steps):
                if mode == "hisparse":
                    events[step][0].record()
                    hisparse_step(step)
                    events[step][1].record()
                else:
                    load(mode, step)
                    events[step][0].record()
                    run(mode, step)
                    events[step][1].record()
                    if mode in misses:
                        uniq[step] = ours[mode][0]._host_batch_buffers[5][1]
            torch.cuda.synchronize(device)
            for step in range(steps):
                samples[mode][step].append(events[step][0].elapsed_time(events[step][1]) * 1000)
            if mode in misses:
                for step, u in enumerate(uniq.tolist()):
                    misses[mode][step].append(u)

    out = dict(requests=requests, mtp=mtp, query_rows=rows, steps=steps, correctness_max_abs_error=errors,
               far_refs=[p["far_refs"] for p in plan], far_unique_uncached=[p["far_unique_step"] for p in plan],
               cached_misses={m: [max(v) for v in vs] for m, vs in misses.items()},
               per_step={m: [summarize(v) for v in vs] for m, vs in samples.items()})
    tail = slice(steps // 2, steps)
    line = " ".join(
        f"{m}={sum(s['median_us'] for s in out['per_step'][m][tail]) / len(out['per_step'][m][tail]):.1f}"
        for m in args.modes)
    print(f"tier={args.tier} requests={requests} mtp={mtp} rows={rows} steady(mean of last half, us): {line} "
          f"misses={ {m: v[-1] for m, v in out['cached_misses'].items()} } unique_uncached={out['far_unique_uncached'][-1]}",
          flush=True)
    del ours, hisparse, query, rope, q_cat, output
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
