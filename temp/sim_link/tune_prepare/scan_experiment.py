"""Use a single warp only for row-prefix scans in the small staged kernel instance."""
import difflib
from pathlib import Path
import subprocess
import os

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
base = (dest / "base.cuh").read_text()
assert header.read_text() == base


def candidate(width=None):
    source = base.replace("template <uint32_t THREADS, typename Value>\n__device__ __forceinline__ uint32_t host_batch_block_scan(",
                          "template <uint32_t THREADS, bool SMALL, typename Value>\n__device__ __forceinline__ uint32_t host_batch_block_scan(", 1)
    old = "  __shared__ uint32_t warp_sums[THREADS / 32], carry;\n  if (tid == 0) carry = 0;"
    new = '''  __shared__ uint32_t warp_sums[THREADS / 32], carry;
  if constexpr (SMALL) {
    if (tid < 32) {
      const uint32_t count = tid < n ? value(tid) : 0;
      uint32_t sum = count;
#pragma unroll
      for (uint32_t offset = 1; offset < 32; offset *= 2) {
        const uint32_t other = __shfl_up_sync(0xffffffff, sum, offset);
        if (tid >= offset) sum += other;
      }
      if (tid < n) out[tid] = sum - count;
      if (tid == 31) carry = sum;
    }
    __syncthreads();
    return carry;
  }
  if (tid == 0) carry = 0;'''
    assert old in source
    source = source.replace(old, new, 1)
    source = source.replace("template <typename KTraits, typename Params>\n__device__ __forceinline__ uint32_t resolve_host_batch_slots(",
                            "template <typename KTraits, bool SMALL, typename Params>\n__device__ __forceinline__ uint32_t resolve_host_batch_slots(", 1)
    assert source.count("host_batch_block_scan<THREADS>(") == 2
    source = source.replace("host_batch_block_scan<THREADS>(", "host_batch_block_scan<THREADS, SMALL>(")
    source = source.replace("resolve_host_batch_slots<KTraits>(params, smem)",
                            "resolve_host_batch_slots<KTraits, ROW_MERGE>(params, smem)", 1)
    if width:
        source = source.replace("template <typename KTraits, typename Params>\n__device__ __forceinline__ void prepare_host_batch_rows(",
                                "template <typename KTraits, bool SMALL, typename Params>\n__device__ __forceinline__ void prepare_host_batch_rows(", 1)
        first = source.index("void prepare_host_batch_rows(")
        end = source.index("// prepare_host_batch_rows without staging", first)
        body = source[first:end].replace("constexpr uint32_t PER_THREAD = 8;", f"constexpr uint32_t PER_THREAD = SMALL ? {width} : 8;", 1)
        source = source[:first] + body + source[end:]
        source = source.replace("prepare_host_batch_rows<KTraits>(params, epoch + 1)",
                                "prepare_host_batch_rows<KTraits, ROW_MERGE>(params, epoch + 1)", 1)
    return source


try:
    for label, width in (("warp_scan", None), ("compact", 1)):
        source = candidate(width)
        (dest / f"{label}.patch").write_text("".join(difflib.unified_diff(
            base.splitlines(True), source.splitlines(True),
            fromfile="a/include/flashinfer/attention/mla_host_batch.cuh",
            tofile="b/include/flashinfer/attention/mla_host_batch.cuh", n=0)))
        header.write_text(source)
        subprocess.run(["python3", str(dest / "screen.py"), label], check=True)
finally:
    header.write_text(base)
print("scan experiment finished; base restored", flush=True)
