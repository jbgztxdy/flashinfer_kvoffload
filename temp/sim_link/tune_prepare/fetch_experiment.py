"""Small-kernel-only compact fetch rings; large-kernel fetch stays unchanged."""
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


def candidate(rows):
    start = base.index("template <typename KTraits, typename Params>\n__device__ __forceinline__ void fetch_host_batch_pages(")
    end = base.index("// Requires the combined [page, 576] layout", start)
    fetch = base[start:end].replace("void fetch_host_batch_pages(", "void fetch_host_batch_small_pages(", 1)
    fetch = fetch.replace("CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW", f"CHUNK_VECS = {rows} * VECS_PER_ROW", 1)
    fetch = fetch.replace("sizeof(typename KTraits::SharedStorage) >= 2 * CHUNK_VECS * 16 ? 2 : 1;",
                          "sizeof(typename KTraits::SharedStorage) >= 4 * CHUNK_VECS * 16 ? 4 :\n"
                          "      sizeof(typename KTraits::SharedStorage) >= 2 * CHUNK_VECS * 16 ? 2 : 1;", 1)
    fetch = fetch.replace("  static_assert(CHUNK_VECS % KTraits::NUM_THREADS == 0);\n", "")
    fetch = fetch.replace("CHUNK_VECS / KTraits::NUM_THREADS", "ceil_div(CHUNK_VECS, KTraits::NUM_THREADS)")
    fetch = fetch.replace("FAR_EMU_INIT(2)", "FAR_EMU_INIT(BUFFERS)")
    fetch = fetch.replace("    FAR_EMU_TILE(2, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));",
                          "    if (chunk * rows < count) {\n"
                          "      FAR_EMU_TILE(BUFFERS, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));\n"
                          "    }")
    fetch = fetch.replace("  if (chunk * rows < count) issue(chunk, 0);",
                          "#pragma unroll\n  for (uint32_t k = 0; k < BUFFERS - 1; ++k) issue(chunk + k * params.fetch_ctas, k);")
    old = '''    const uint32_t next = chunk + params.fetch_ctas;
    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);'''
    new = '''    issue(chunk + (BUFFERS - 1) * params.fetch_ctas, (k + BUFFERS - 1) % BUFFERS);
    cp_async::wait_group<BUFFERS - 1>();
    FAR_EMU_WAIT(BUFFERS, k % BUFFERS);'''
    assert old in fetch
    fetch = fetch.replace(old, new, 1).replace("BUFFERS == 2 ? k % 2 : 0", "k % BUFFERS")
    fetch = fetch.replace("    if (BUFFERS == 1 && next * rows < count) issue(next, 0);", "")
    fetch = fetch.replace("  // Deferred staged chunks:", "  cp_async::wait_group<0>();\n  // Deferred staged chunks:", 1)
    source = base[:start] + fetch + base[start:]
    # Compact buffers require rows <= their physical stride, including fetch_ctas <= 8.
    first = source.index("  if constexpr (ROW_MERGE) {", source.index("uint32_t host_batch_fetch_rows("))
    last = source.index("  } else {", first)
    body = source[first:last].replace(" : kHostBatchFetchRows", f" : {rows}u")
    source = source[:first] + body + source[last:]
    call = "  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);"
    assert call in source
    source = source.replace(call, '''  if (unique > 0 && from_back && !host_batch_writes_back(params)) {
    if constexpr (ROW_MERGE) fetch_host_batch_small_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
    else fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  }''', 1)
    return source


try:
    for rows in (32, 16):
        label = f"ring{rows}"
        source = candidate(rows)
        (dest / f"{label}.patch").write_text("".join(difflib.unified_diff(
            base.splitlines(True), source.splitlines(True),
            fromfile="a/include/flashinfer/attention/mla_host_batch.cuh",
            tofile="b/include/flashinfer/attention/mla_host_batch.cuh", n=0)))
        header.write_text(source)
        subprocess.run(["python3", str(dest / "screen.py"), label], check=True)
finally:
    header.write_text(base)
print("fetch experiment finished; base restored", flush=True)
