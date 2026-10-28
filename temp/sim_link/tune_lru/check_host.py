"""Alternate old/new Host timing to resolve the small-batch variation seen in the full suite."""
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
baseline = subprocess.check_output(["git", "show", "d92e7e83:" + str(header)])
dest = Path("temp/sim_link/tune_lru")
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(root),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
env.pop("FLASHINFER_FAR_EMU", None)
try:
    for i, (label, source) in enumerate((("base", baseline), ("candidate", candidate),
                                        ("candidate", candidate), ("base", baseline))):
        header.write_bytes(source)
        for fetch in (8, 16):
            output = dest / f"host_confirm_{i}_{label}_f{fetch}.json"
            assert not output.exists()
            cmd = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "temp/sim_link/bench_tune.py",
                   "--tier", "host", "--mtp", "8", "--requests", "1", "4", "16", "64", "--passes", "9",
                   "--fetch-ctas", str(fetch), "--modes", "ours_stage", "--output", str(output)]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            d = json.loads(output.read_text())
            print(output.name, [(r["requests"], sum(s["median_us"] for s in r["per_step"]["ours_stage"][r["steps"]//2:]) /
                  (r["steps"]-r["steps"]//2)) for r in d["results"]], flush=True)
finally:
    header.write_bytes(candidate)
print("Alternating Host check complete; candidate restored", flush=True)
