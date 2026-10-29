"""Keep each candidate and restore the base after profiling and controlled screens."""
import difflib
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
base = (dest / "base.cuh").read_text()
assert header.read_text() == base


def save(label, source):
    (dest / f"{label}.patch").write_text("".join(difflib.unified_diff(
        base.splitlines(True), source.splitlines(True),
        fromfile="a/include/flashinfer/attention/mla_host_batch.cuh",
        tofile="b/include/flashinfer/attention/mla_host_batch.cuh", n=0)))
    header.write_text(source)


def preparation(width):
    source = base.replace("template <typename KTraits, typename Params>\n__device__ __forceinline__ void prepare_host_batch_rows(",
                          "template <typename KTraits, bool SMALL, typename Params>\n__device__ __forceinline__ void prepare_host_batch_rows(", 1)
    first = source.index("void prepare_host_batch_rows(")
    end = source.index("// prepare_host_batch_rows without staging", first)
    body = source[first:end].replace("constexpr uint32_t PER_THREAD = 8;",
                                     f"constexpr uint32_t PER_THREAD = SMALL ? {width} : 8;", 1)
    source = source[:first] + body + source[end:]
    return source.replace("prepare_host_batch_rows<KTraits>(params, epoch + 1)",
                          "prepare_host_batch_rows<KTraits, ROW_MERGE>(params, epoch + 1)", 1)


def profile_source():
    helper = '''__device__ __forceinline__ unsigned long long hb_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define HB_MARK(k) do { if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { \\
  (reinterpret_cast<unsigned long long*>(params.partial_o + static_cast<int64_t>(params.partial_capacity) * 64 * 512) - 4096)[blockIdx.y * 16 + (k)] = hb_now(); } } while (0)
'''
    source = base.replace("// Staging slots copied and published together.", helper + "\n// Staging slots copied and published together.", 1)
    source = source.replace("const uint32_t epoch = params.state[kHostBatchEpoch];", "HB_MARK(0);\n  const uint32_t epoch = params.state[kHostBatchEpoch];", 1)
    source = source.replace("    cg::this_grid().sync();\n    gaps =", "    HB_MARK(1);\n    cg::this_grid().sync();\n    HB_MARK(2);\n    gaps =", 1)
    source = source.replace("    if (gaps) compact_host_batch_rows<KTraits>(params);", "    HB_MARK(3);\n    if (gaps) compact_host_batch_rows<KTraits>(params);", 1)
    source = source.replace("    cg::this_grid().sync();\n    // Claim counts", "    HB_MARK(4);\n    cg::this_grid().sync();\n    // Claim counts", 1)
    source = source.replace("  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);", "  HB_MARK(5);\n  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);\n  HB_MARK(6);", 1)
    source = source.replace("  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);", "  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);\n  HB_MARK(7);", 1)
    source = source.replace("    if constexpr (!ROW_MERGE) cg::this_grid().sync();", "    if constexpr (!ROW_MERGE) cg::this_grid().sync();\n    HB_MARK(8);", 1)
    source = source.replace("  // The last CTA to leave resets", "  HB_MARK(9);\n  // The last CTA to leave resets", 1)
    source = source.replace("      run_host_batch_lru<KTraits>(params, smem, item, scratch[kHostBatchScratchEpoch], unique);", "      HB_MARK(10);\n      run_host_batch_lru<KTraits>(params, smem, item, scratch[kHostBatchScratchEpoch], unique);\n      HB_MARK(11);", 1)
    return source


env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(root),
           TMPDIR=str(root / "temp"), FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU",
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu_prepare_profile"))
try:
    save("profile", profile_source())
    for mtp in (1, 8):
        with (dest / f"profile_mtp{mtp}.log").open("w") as log:
            subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                            str(dest / "profile.py"), "--requests", "4", "--mtp", str(mtp),
                            "--emu", "100:20000", "--output", str(dest / f"profile_mtp{mtp}.json")],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("profile complete", mtp, flush=True)
    for width in (1, 2, 4):
        label = f"prepare{width}"
        save(label, preparation(width))
        subprocess.run(["python3", str(dest / "screen.py"), label], check=True)
finally:
    header.write_text(base)
print("experiment finished; base restored", flush=True)
