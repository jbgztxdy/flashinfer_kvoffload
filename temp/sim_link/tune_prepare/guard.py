"""Expose the measured small-prepare path explicitly; the default preprocesses to the baseline."""
from pathlib import Path
import difflib

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
base = (dest / "base.cuh").read_text()
source = (dest / "isolated.cuh").read_text()


def function(name):
    pos = source.index(name + "(")
    first = source.rfind("template <", 0, pos)
    brace = source.index("{", pos)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[first:end] + "\n\n"


prepare = function("prepare_host_batch_rows").replace(
    "template <typename KTraits, bool SMALL, typename Params>", "template <typename KTraits, typename Params>", 1)
prepare = prepare.replace("prepare_host_batch_rows(", "prepare_host_batch_small_rows(", 1)
prepare = prepare.replace("PER_THREAD = SMALL ? 1 : 8", "PER_THREAD = 1", 1)
helpers = prepare + function("host_batch_small_scan") + function("resolve_host_batch_small_slots")
marker = "template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__"
assert marker in base
guarded = base.replace(marker, "// Optional short-segment prepare and single-warp scans for high-latency tuning.\n"
                       "// Keep the default translation unit identical to the measured baseline.\n"
                       "#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE\n" + helpers + "#endif\n\n" + marker, 1)
guarded = guarded.replace("      prepare_host_batch_rows<KTraits>(params, epoch + 1);", '''#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif''', 1)
guarded = guarded.replace("    unique = resolve_host_batch_slots<KTraits>(params, smem);", '''#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif''', 1)
(dest / "guarded.patch").write_text("".join(difflib.unified_diff(
    base.splitlines(True), guarded.splitlines(True), fromfile="a/include/flashinfer/attention/mla_host_batch.cuh",
    tofile="b/include/flashinfer/attention/mla_host_batch.cuh", n=0)))
(dest / "guarded.cuh").write_text(guarded)
header.write_text(guarded)
print("explicit SMALL_PREPARE option installed; default calls the original helpers")
