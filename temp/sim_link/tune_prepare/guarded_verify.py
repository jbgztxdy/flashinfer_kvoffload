"""Validate the opt-in feature, then compare enabled/default builds without changing source."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
assert header.read_bytes() == (dest / "guarded.cuh").read_bytes()
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", TMPDIR=str(root / "temp"),
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root),
           FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_HOST_BATCH_SMALL_PREPARE",
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_prepare_guarded_normal"))
with (dest / "pytest_guarded.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest", "-q",
                    "-p", "no:cacheprovider", "-p", "jitpaths", "tests/attention/test_mla_host_batch.py"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("guarded opt-in full pytest passed", flush=True)
subprocess.run(["python3", str(dest / "screen.py"), "enabled", "--small-prepare", "--full"], check=True)
subprocess.run(["python3", str(dest / "screen.py"), "default", "--full"], check=True)
subprocess.run(["python3", str(dest / "screen.py"), "enabled_repeat", "--small-prepare"], check=True)
print("guarded verification complete", flush=True)
