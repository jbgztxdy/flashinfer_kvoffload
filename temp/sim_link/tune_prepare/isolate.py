"""Keep original shared-memory helpers on the general path; specialize only small staged scans."""
from pathlib import Path
import difflib

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
base = (dest / "base.cuh").read_text()
source = (dest / "accepted.cuh").read_text()
assert header.read_text() == source


def section(text, start, end):
    a = text.index(start)
    b = text.index(end, a)
    return a, b, text[a:b]


scan_start = "// Exclusive scan of value(i) for i < n"
scan_end = "// After the grid barrier: turn each Host entry"
resolve_end = "// After the grid barrier, when prepare dropped padding"
a, b, scan = section(source, scan_start, scan_end)
small_scan = scan.replace("template <uint32_t THREADS, bool SMALL, typename Value>",
                          "template <uint32_t THREADS, typename Value>", 1)
small_scan = small_scan.replace("host_batch_block_scan(", "host_batch_small_scan(", 1)
start = small_scan.index("  if constexpr (SMALL) {")
stop = small_scan.index("  if (tid == 0) carry = 0;", start)
body = small_scan[start:stop].removeprefix("  if constexpr (SMALL) {\n").removesuffix("  }\n")
small_scan = small_scan[:start] + body + "}\n\n"
original_scan = section(base, scan_start, scan_end)[2]
source = source[:a] + original_scan + small_scan + source[b:]

a, b, resolve = section(source, scan_end, resolve_end)
small_resolve = resolve.replace("template <typename KTraits, bool SMALL, typename Params>",
                                "template <typename KTraits, typename Params>", 1)
small_resolve = small_resolve.replace("resolve_host_batch_slots(", "resolve_host_batch_small_slots(", 1)
small_resolve = small_resolve.replace("host_batch_block_scan<THREADS, SMALL>", "host_batch_small_scan<THREADS>")
original_resolve = section(base, scan_end, resolve_end)[2]
source = source[:a] + original_resolve + small_resolve + source[b:]

fetch_start = "// Copy staged pages Host -> HBM in slot order"
fetch_end = "// Requires the combined [page, 576] layout"
a, b, _ = section(source, fetch_start, fetch_end)
original_fetch = section(base, fetch_start, fetch_end)[2]
source = source[:a] + original_fetch + source[b:]

rows_start = "// Staging slots per fetch chunk"
rows_end = "template <typename Params>\n__device__ __forceinline__ void bind_host_batch_staged_slot"
a, b, _ = section(source, rows_start, rows_end)
original_rows = section(base, rows_start, rows_end)[2]
source = source[:a] + original_rows + source[b:]

source = source.replace("    unique = resolve_host_batch_slots<KTraits, ROW_MERGE>(params, smem);",
                        "    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);\n"
                        "    else unique = resolve_host_batch_slots<KTraits>(params, smem);", 1)
a = source.index("  // Compact chunks help the high-latency experiments")
b = source.index("  if (tid == 0) {\n    scratch[kHostBatchScratchEpoch]", a)
source = source[:a] + "  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);\n" + source[b:]
source = source.replace("fetch_host_batch_pages<KTraits, Params, COMPACT_FETCH>(params, smem, epoch, unique, fetch_rows)",
                        "fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows)", 1)
assert "COMPACT" not in source
(dest / "isolated.patch").write_text("".join(difflib.unified_diff(
    base.splitlines(True), source.splitlines(True), fromfile="a/include/flashinfer/attention/mla_host_batch.cuh",
    tofile="b/include/flashinfer/attention/mla_host_batch.cuh", n=0)))
(dest / "isolated.cuh").write_text(source)
header.write_text(source)
print("isolated original general helpers; compact pipeline stays in the saved experimental patch")
