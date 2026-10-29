"""Full tests and reversed same-GPU matrices; restore the candidate even after failure."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = root / "include/flashinfer/attention/mla_host_batch.cuh"
final = header.read_bytes()
assert final == (dest / "final.cuh").read_bytes()
base = (dest / "base.cuh").read_bytes()
assert final != base
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", TMPDIR=str(root / "temp"),
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_prepare_normal"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
with (dest / "pytest_final.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest", "-q",
                    "-p", "no:cacheprovider", "-p", "jitpaths", "tests/attention/test_mla_host_batch.py"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("full pytest passed", flush=True)
try:
    subprocess.run(["python3", str(dest / "screen.py"), "final", "--full"], check=True)
    header.write_bytes(base)
    subprocess.run(["python3", str(dest / "screen.py"), "base_full", "--full"], check=True)
    header.write_bytes(final)
    subprocess.run(["python3", str(dest / "screen.py"), "final_repeat"], check=True)
finally:
    header.write_bytes(final)
print("verification finished; final restored", flush=True)
