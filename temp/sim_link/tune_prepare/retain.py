"""Validate the isolated final default and repeat its controlled matrix."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
retained = header.read_bytes()
assert retained == (dest / "isolated.cuh").read_bytes()
(dest / "retained.cuh").write_bytes(retained)
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", TMPDIR=str(root / "temp"),
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_prepare_normal"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
with (dest / "pytest_retained.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest", "-q",
                    "-p", "no:cacheprovider", "-p", "jitpaths", "tests/attention/test_mla_host_batch.py"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("retained full pytest passed", flush=True)
subprocess.run(["python3", str(dest / "screen.py"), "retained", "--full"], check=True)
subprocess.run(["python3", str(dest / "screen.py"), "retained_repeat"], check=True)
print("retained verification complete", flush=True)
