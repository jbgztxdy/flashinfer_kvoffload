"""Check new LRU changes with the previous round's recommended planner settings."""
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
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(root),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"), FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
try:
    for label, source in (("tuned_candidate", candidate), ("tuned_base", baseline)):
        header.write_bytes(source)
        for name, mtp, reqs, fetch, items in (
                ("mtp8_f24", 8, [1, 2, 4, 16, 64, 128], 24, 64),
                ("mtp8_h256", 8, [1, 2, 4], 16, 256),
                ("mtp1_f32", 1, [1, 4, 16, 128], 32, 64)):
            output = dest / f"{label}_{name}.json"
            assert not output.exists()
            cmd = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "temp/sim_link/bench_tune.py",
                   "--tier", "peer", "--mtp", str(mtp), "--requests", *map(str, reqs), "--passes", "7",
                   "--fetch-ctas", str(fetch), "--host-item-entries", str(items),
                   "--modes", "ours_stage", "memcpy_lru", "direct_lru", "--emu", "100:20000", "--output", str(output)]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("verified", output.name, flush=True)
finally:
    header.write_bytes(candidate)
print("Tuned-settings check complete; candidate restored", flush=True)
