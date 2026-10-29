"""Build the compact prepare/scan + 32-row ring with the original large-path fetch intact."""
import difflib
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
header_name = "include/flashinfer/attention/mla_host_batch.cuh"
header = root / header_name
base = (dest / "base.cuh").read_text()
assert header.read_text() == base
with tempfile.TemporaryDirectory(dir=root / "temp", prefix="combine_prepare_") as directory:
    target = Path(directory) / header_name
    target.parent.mkdir(parents=True)
    target.write_text(base)
    subprocess.run(["patch", "--silent", "-p1", "-i", str(dest / "compact.patch")], cwd=directory, check=True)
    source = target.read_text()

start = source.index("template <typename KTraits, typename Params>\n__device__ __forceinline__ void fetch_host_batch_pages(")
end = source.index("// Requires the combined [page, 576] layout", start)
fetch = source[start:end].replace("template <typename KTraits, typename Params>",
                                  "template <typename KTraits, typename Params, bool COMPACT>", 1)
fetch = fetch.replace("CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW", "CHUNK_VECS = (COMPACT ? 32 : kHostBatchFetchRows) * VECS_PER_ROW", 1)
fetch = fetch.replace("sizeof(typename KTraits::SharedStorage) >= 2 * CHUNK_VECS * 16 ? 2 : 1;",
                      "COMPACT && sizeof(typename KTraits::SharedStorage) >= 4 * CHUNK_VECS * 16 ? 4 :\n"
                      "      sizeof(typename KTraits::SharedStorage) >= 2 * CHUNK_VECS * 16 ? 2 : 1;", 1)
fetch = fetch.replace("  FAR_EMU_INIT(2);", "  constexpr uint32_t EMU_STAGES = COMPACT ? BUFFERS : 2;\n  FAR_EMU_INIT(EMU_STAGES);", 1)
old = "    FAR_EMU_TILE(2, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));"
assert old in fetch
fetch = fetch.replace(old, '''    if constexpr (COMPACT) {
      if (chunk * rows < count) {
        FAR_EMU_TILE(EMU_STAGES, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));
      }
    } else {
      FAR_EMU_TILE(EMU_STAGES, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));
    }''', 1)
fetch = fetch.replace("  if (chunk * rows < count) issue(chunk, 0);", '''  if constexpr (COMPACT) {
#pragma unroll
    for (uint32_t k = 0; k < BUFFERS - 1; ++k) issue(chunk + k * params.fetch_ctas, k);
  } else {
    if (chunk * rows < count) issue(chunk, 0);
  }''', 1)
old = '''    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);'''
assert old in fetch
fetch = fetch.replace(old, '''    if constexpr (COMPACT) {
      issue(chunk + (BUFFERS - 1) * params.fetch_ctas, (k + BUFFERS - 1) % BUFFERS);
      cp_async::wait_group<BUFFERS - 1>();
    } else {
      if (BUFFERS == 2 && next * rows < count) {
        issue(next, (k + 1) % 2);
        cp_async::wait_group<1>();
      } else {
        cp_async::wait_group<0>();
      }
    }
    FAR_EMU_WAIT(EMU_STAGES, COMPACT ? k % BUFFERS : (BUFFERS == 2 ? k % 2 : 0));''', 1)
fetch = fetch.replace("const uint32_t buf = BUFFERS == 2 ? k % 2 : 0;", "const uint32_t buf = COMPACT ? k % BUFFERS : (BUFFERS == 2 ? k % 2 : 0);", 1)
fetch = fetch.replace("    if (BUFFERS == 1 && next * rows < count) issue(next, 0);", "    if constexpr (!COMPACT) {\n      if (BUFFERS == 1 && next * rows < count) issue(next, 0);\n    }", 1)
fetch = fetch.replace("  // Deferred staged chunks:", "  if constexpr (COMPACT) cp_async::wait_group<0>();\n  // Deferred staged chunks:", 1)
source = source[:start] + fetch + source[end:]
first = source.index("  if constexpr (ROW_MERGE) {", source.index("uint32_t host_batch_fetch_rows("))
last = source.index("  } else {", first)
source = source[:first] + source[first:last].replace(" : kHostBatchFetchRows", " : 32u") + source[last:]
source = source.replace("fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows)",
                        "fetch_host_batch_pages<KTraits, Params, ROW_MERGE>(params, smem, epoch, unique, fetch_rows)", 1)
(dest / "combined.patch").write_text("".join(difflib.unified_diff(
    base.splitlines(True), source.splitlines(True), fromfile="a/" + header_name,
    tofile="b/" + header_name, n=0)))
header.write_text(source)
print("combined candidate installed", flush=True)
