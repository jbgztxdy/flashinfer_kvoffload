"""Reverse-order physical Host check for a slower in-place cell despite unchanged code path."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
accepted = header.read_bytes()
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", TMPDIR=str(root / "temp"),
           PYTHONPATH=str(root), FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_prepare_normal"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
env.pop("FLASHINFER_FAR_EMU", None)
try:
    for label, source in (("accepted_host_recheck", accepted), ("base_host_recheck", (dest / "base.cuh").read_bytes())):
        if header.read_bytes() != source:
            header.write_bytes(source)
        output = dest / f"{label}_proxy_host_mtp1.json"
        assert not output.exists()
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                            "temp/sim_link/bench_tune.py", "--tier", "host", "--peer", "0", "--mtp", "1",
                            "--requests", "1", "4", "16", "128", "--passes", "7", "--fetch-ctas", "16",
                            "--host-item-entries", "64", "--modes", "ours_stage", "ours_inplace", "floor_fa2",
                            "--output", str(output)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("verified", output.name, flush=True)
finally:
    header.write_bytes(accepted)
print("Host recheck finished; accepted restored", flush=True)
