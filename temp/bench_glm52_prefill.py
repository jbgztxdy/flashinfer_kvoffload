"""Purpose: Benchmark GLM-5.2-shaped FlashInfer MLA prefill with chunked queries.
Usage: CUDA_VISIBLE_DEVICES=0 python temp/bench_glm52_prefill.py --context-tokens 414848 --chunk-size 4096
"""

import argparse
import gc
import math
import pathlib

import torch

import flashinfer.jit.env as jit_env

# Source checkouts keep these directories at repository root; wheels package them
# below flashinfer/data instead.
_repo_root = pathlib.Path(__file__).resolve().parents[1]
jit_env.FLASHINFER_DATA = _repo_root
jit_env.FLASHINFER_CSRC_DIR = _repo_root / "csrc"
jit_env.FLASHINFER_INCLUDE_DIR = _repo_root / "include"
jit_env.CUTLASS_INCLUDE_DIRS = [
    _repo_root / "3rdparty/cutlass/include",
    _repo_root / "3rdparty/cutlass/tools/util/include",
]
jit_env.SPDLOG_INCLUDE_DIR = _repo_root / "3rdparty/spdlog/include"

from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


NUM_HEADS = 64
HEAD_DIM_QK = 256
HEAD_DIM_VO = 256
DTYPE = torch.bfloat16


def _event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-tokens", type=int, default=414_848)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("fa2", "fa3"), default="fa2")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.context_tokens <= 0 or args.chunk_size <= 0:
        raise ValueError("context-tokens and chunk-size must be positive")

    device = torch.device(args.device)
    kv_tokens = args.context_tokens + args.chunk_size
    qo_indptr = torch.tensor([0, args.chunk_size], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_tokens], dtype=torch.int32, device=device)

    # This is the post-projection MHA view used by FlashInfer's MLA prefill backend.
    q = torch.empty(
        (args.chunk_size, NUM_HEADS, HEAD_DIM_QK), dtype=DTYPE, device=device
    )
    k = torch.empty((kv_tokens, NUM_HEADS, HEAD_DIM_QK), dtype=DTYPE, device=device)
    v = torch.empty((kv_tokens, NUM_HEADS, HEAD_DIM_VO), dtype=DTYPE, device=device)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=args.backend
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        head_dim_qk=HEAD_DIM_QK,
        head_dim_vo=HEAD_DIM_VO,
        causal=True,
        sm_scale=1.0 / math.sqrt(HEAD_DIM_QK),
        q_data_type=DTYPE,
        o_data_type=DTYPE,
    )

    for _ in range(args.warmup):
        wrapper.run(q, k, v)
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = wrapper.run(q, k, v)
        end.record()
        end.synchronize()
        timings.append(_event_elapsed_ms(start, end))

    timings.sort()
    median_ms = timings[len(timings) // 2]
    # Two GEMMs (QK and PV), reported as a useful attention-only proxy.
    flops = 2.0 * args.chunk_size * kv_tokens * NUM_HEADS * HEAD_DIM_QK
    tflops = flops / (median_ms * 1e9)
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"context_tokens={args.context_tokens} kv_tokens={kv_tokens}")
    print(f"chunk_size={args.chunk_size} heads={NUM_HEADS} head_dim={HEAD_DIM_QK}")
    print(f"backend={args.backend} dtype={DTYPE} output_shape={tuple(out.shape)}")
    print(f"timings_ms={','.join(f'{x:.3f}' for x in timings)}")
    print(f"median_ms={median_ms:.3f} attention_tflops={tflops:.2f}")
    print(
        f"allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.2f} "
        f"reserved_gib={torch.cuda.memory_reserved(device) / 2**30:.2f}"
    )

    del out, wrapper, workspace, q, k, v
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
