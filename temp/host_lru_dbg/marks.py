# Purpose: add (apply) or remove (strip) temporary %globaltimer phase marks in mla_host_batch.cuh.
# Slots: 0 start, 1 prepare done, 2 after barrier 1, 3 resolve done, 4 LRU done (in place),
# 5 items start, 6 fetch done, 7 items done, 8 after merge barrier, 9 end.
import re, sys
p = "include/flashinfer/attention/mla_host_batch.cuh"
s = open(p).read()
MACRO = """__device__ __forceinline__ unsigned long long hb_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define HB_MARK(k) do { if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { \\
  (reinterpret_cast<unsigned long long*>(params.partial_o + static_cast<int64_t>(params.partial_capacity) * 64 * 512) - 4096)[blockIdx.y * 16 + (k)] = hb_now(); } } while (0)
"""
SITES = [
    ("  const uint32_t epoch = params.state[kHostBatchEpoch];\n", "after", 0),
    ("    cg::this_grid().sync();\n    gaps = params.stage", "before", 1),
    ("    gaps = params.stage", "before", 2),
    ("    // In place, victims are chosen now", "before", 3),
    ("  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;", "before", 4),
    ("  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages", "before", 5),
    ("  run_host_batch_items<KTraits>(params, storage, smem, scratch, from_back, unique);", "before", 6),
    ("  // Uniform over the grid, so every CTA takes the same branch.\n  if (params.full_rows < params.num_rows) {", "before", 7),
    ("    // 32 features per thread fill the grid", "before", 8),
    ("  // The last CTA to leave resets the shared counters", "before", 9),
]
if sys.argv[1] == "apply":
    assert "HB_MARK" not in s
    s = s.replace("namespace mla {\n", "namespace mla {\n" + MACRO, 1)
    for anchor, where, k in SITES:
        assert s.count(anchor) == 1, anchor
        mark = f"  HB_MARK({k});\n"
        s = s.replace(anchor, mark + anchor if where == "before" else anchor + mark)
else:
    s = s.replace(MACRO, "")
    s = re.sub(r"  HB_MARK\(\d+\);\n", "", s)
    assert "HB_MARK" not in s
open(p, "w").write(s)
