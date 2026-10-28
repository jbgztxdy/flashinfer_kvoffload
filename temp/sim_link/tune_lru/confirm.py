"""Repeat candidate then baseline in reverse order; always restore the candidate source."""
import argparse
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", type=Path, default=Path("temp/sim_link/tune_lru"))
dest = ap.parse_args().output_dir
dest.mkdir(parents=True, exist_ok=True)
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
baseline = subprocess.check_output(["git", "show", "d92e7e83:" + str(header)])
try:
    subprocess.run(["python3", "temp/sim_link/tune_lru/run.py", "candidate_repeat", "--suite", "full", "--passes", "7", "--output-dir", str(dest)], check=True)
    header.write_bytes(baseline)
    subprocess.run(["python3", "temp/sim_link/tune_lru/run.py", "base_repeat", "--suite", "full", "--passes", "7", "--output-dir", str(dest)], check=True)
finally:
    header.write_bytes(candidate)
print("Reverse-order confirmation complete; candidate restored", flush=True)

env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(root),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
env.pop("FLASHINFER_FAR_EMU", None)
for mtp, context, requests, captures, steps in (
        (8, 65536, [1, 2, 4, 16, 64, 128], None, None),
        (1, 65536, [1, 4, 16, 128], None, None),
        (1, 8192, [1, 4], "temp/glm_trace/validation8k/doc0.pt", 256)):
    name = f"floor_ctx{context}_mtp{mtp}"
    output = dest / (name + ".json")
    assert not output.exists()
    cmd = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "temp/h800/bench_rounds.py",
           "--tier", "peer", "--mtp", str(mtp), "--context", str(context),
           "--requests", *map(str, requests), "--passes", "7", "--modes", "floor_fa2", "--output", str(output)]
    if captures:
        cmd += ["--captures", captures, "--steps", str(steps)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("verified", name, flush=True)
