"""Experiment-only overrides of existing Host MLA planner controls.

All other arguments, correctness checks and timings come from bench_rounds.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h800"))
import bench_rounds

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--host-item-entries", type=int, default=None)
parser.add_argument("--host-split", type=int, choices=(0, 1), default=None)
parser.add_argument("--grid-ctas", type=int, default=None)
parser.add_argument("--kv-chunks", type=int, default=None)
overrides, remaining = parser.parse_known_args()
if overrides.kv_chunks is not None:
    if "-DFLASHINFER_HOST_BATCH_TUNE" not in os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""):
        raise SystemExit("--kv-chunks needs the experiment-only FLASHINFER_HOST_BATCH_TUNE build")
    if 'std::getenv("FLASHINFER_HOST_BATCH_KV_CHUNKS")' not in Path(
            "include/flashinfer/attention/mla_host_batch.cuh").read_text():
        raise SystemExit("--kv-chunks needs the saved planner patch applied")
    os.environ["FLASHINFER_HOST_BATCH_KV_CHUNKS"] = str(overrides.kv_chunks)
if overrides.grid_ctas is not None:
    if "-DFLASHINFER_HOST_BATCH_TUNE" not in os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""):
        raise SystemExit("--grid-ctas needs the experiment-only FLASHINFER_HOST_BATCH_TUNE build")
    if 'std::getenv("FLASHINFER_HOST_BATCH_GRID_LIMIT")' not in Path(
            "include/flashinfer/attention/mla_host_batch.cuh").read_text():
        raise SystemExit("--grid-ctas needs the saved grid_limit.patch applied in a baseline checkout")
    os.environ["FLASHINFER_HOST_BATCH_GRID_LIMIT"] = str(overrides.grid_ctas)
make_wrapper = bench_rounds.make_wrapper


def tuned_wrapper(*args, **kwargs):
    wrapper = make_wrapper(*args, **kwargs)
    if overrides.host_item_entries is not None:
        wrapper.host_item_entries = overrides.host_item_entries
    if overrides.host_split is not None:
        wrapper.host_split = bool(overrides.host_split)
    return wrapper


bench_rounds.make_wrapper = tuned_wrapper
sys.argv = [sys.argv[0]] + remaining
bench_rounds.main()
output = Path(remaining[remaining.index("--output") + 1])
result = json.loads(output.read_text())
result["planner_overrides"] = vars(overrides)
result["extra_cuda_flags"] = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
result["source_sha256"]["temp/sim_link/bench_tune.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result["source_sha256"]["include/flashinfer/attention/far_link_emu.cuh"] = hashlib.sha256(
    Path("include/flashinfer/attention/far_link_emu.cuh").read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=1) + "\n")
