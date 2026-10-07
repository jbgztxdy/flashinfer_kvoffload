"""Purpose: Benchmark GLM-5.2-shaped FlashInfer MLA decode on a paged KV cache.
Usage: CUDA_VISIBLE_DEVICES=0 python temp/bench_glm52_decode.py --kv-tokens 418944
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

from flashinfer.mla import BatchMLAPagedAttentionWrapper


NUM_HEADS = 64
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
DTYPE = torch.bfloat16
PAGE_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-tokens", type=int, default=418_944)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("fa2", "fa3"), default="fa2")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.kv_tokens <= 0 or args.kv_tokens % PAGE_SIZE != 0:
        raise ValueError("kv-tokens must be positive and divisible by page size 64")

    device = torch.device(args.device)
    num_pages = args.kv_tokens // PAGE_SIZE
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([args.kv_tokens], dtype=torch.int32, device=device)

    # Matrix-absorbed MLA decode stores one compressed KV vector and one RoPE
    # vector per token; the output remains [batch, heads, compressed_rank].
    q_nope = torch.empty((1, NUM_HEADS, HEAD_DIM_CKV), dtype=DTYPE, device=device)
    q_pe = torch.empty((1, NUM_HEADS, HEAD_DIM_KPE), dtype=DTYPE, device=device)
    ckv = torch.empty(
        (num_pages, PAGE_SIZE, HEAD_DIM_CKV), dtype=DTYPE, device=device
    )
    kpe = torch.empty(
        (num_pages, PAGE_SIZE, HEAD_DIM_KPE), dtype=DTYPE, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=args.backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=NUM_HEADS,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=PAGE_SIZE,
        causal=False,
        sm_scale=1.0 / math.sqrt(256),
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )

    for _ in range(args.warmup):
        wrapper.run(q_nope, q_pe, ckv, kpe)
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = wrapper.run(q_nope, q_pe, ckv, kpe)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    timings.sort()
    median_ms = timings[len(timings) // 2]
    # QK and PV each process the compressed rank plus the RoPE dimensions.
    flops = 2.0 * NUM_HEADS * (2 * HEAD_DIM_CKV + HEAD_DIM_KPE) * args.kv_tokens
    tflops = flops / (median_ms * 1e9)
    cache_bytes = args.kv_tokens * (HEAD_DIM_CKV + HEAD_DIM_KPE) * DTYPE.itemsize
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"kv_tokens={args.kv_tokens} num_pages={num_pages} page_size={PAGE_SIZE}")
    print(
        f"batch=1 query_tokens=1 heads={NUM_HEADS} "
        f"head_dim_ckv={HEAD_DIM_CKV} head_dim_kpe={HEAD_DIM_KPE}"
    )
    print(f"backend={args.backend} dtype={DTYPE} output_shape={tuple(out.shape)}")
    print(f"kv_cache_gib={cache_bytes / 2**30:.2f}")
    print(f"timings_ms={','.join(f'{x:.3f}' for x in timings)}")
    print(f"median_ms={median_ms:.3f} attention_tflops={tflops:.2f}")
    print(
        f"allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.2f} "
        f"reserved_gib={torch.cuda.memory_reserved(device) / 2**30:.2f}"
    )

    del out, wrapper, workspace, q_nope, q_pe, ckv, kpe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
