# Purpose: Verify and benchmark top-k MLA decode via one-token FlashInfer pages.
# Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python temp/bench_glm52_sparse_decode.py --kv-tokens 418944 --topk 2048

import argparse
import math
import statistics

import torch

# Reuse the source-checkout JIT paths from the dense decode benchmark.
from bench_glm52_decode import BatchMLAPagedAttentionWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-tokens", type=int, default=418_944)
    parser.add_argument("--topk", type=int, default=2048)
    args = parser.parse_args()
    if not 0 < args.topk <= args.kv_tokens:
        raise ValueError("topk must be positive and no greater than kv-tokens")

    torch.manual_seed(42)
    device = "cuda:0"
    heads, rank, rope_dim = 64, 512, 64
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(256)
    q_nope = torch.randn((1, heads, rank), device=device, dtype=dtype)
    q_pe = torch.randn((1, heads, rope_dim), device=device, dtype=dtype)
    ckv = torch.randn((args.kv_tokens, 1, rank), device=device, dtype=dtype)
    kpe = torch.randn((args.kv_tokens, 1, rope_dim), device=device, dtype=dtype)
    selected = torch.randperm(args.kv_tokens, device=device)[: args.topk]
    kv_indices = selected.to(torch.int32)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, args.topk], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([args.topk], dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=heads,
        head_dim_ckv=rank,
        head_dim_kpe=rope_dim,
        page_size=1,
        # Selected indices already enforce causality for this decode query.
        causal=False,
        sm_scale=scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = torch.empty_like(q_nope)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=out)

    # Only the selected physical token slots contribute to the reference.
    selected_ckv = ckv[selected, 0].float()
    selected_kpe = kpe[selected, 0].float()
    scores = (
        q_nope[0].float() @ selected_ckv.T
        + q_pe[0].float() @ selected_kpe.T
    ) * scale
    reference = (scores.softmax(dim=-1) @ selected_ckv).unsqueeze(0)
    torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=5e-3)
    max_error = (out.float() - reference).abs().max().item()

    for _ in range(3):
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    torch.cuda.synchronize()
    eager_times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
        end.record()
        end.synchronize()
        eager_times.append(start.elapsed_time(end) * 1000)

    # Capture repeated runs to keep host launch gaps out of the short timing.
    repeats = 100
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    graph.replay()
    torch.cuda.synchronize()
    graph_times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        graph_times.append(start.elapsed_time(end) * 1000 / repeats)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"context_tokens={args.kv_tokens} attended_tokens={args.topk} page_size=1")
    print(f"batch=1 heads={heads} rank={rank} rope_dim={rope_dim} dtype={dtype}")
    print(f"correctness=passed max_abs_error={max_error:.6f}")
    print(f"output_shape={tuple(out.shape)}")
    print(f"eager_median_us={statistics.median(eager_times):.3f}")
    print(f"graph_median_us={statistics.median(graph_times):.3f}")


if __name__ == "__main__":
    main()
