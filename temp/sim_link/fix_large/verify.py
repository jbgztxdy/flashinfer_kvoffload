"""Test both dispatch paths, then reverse-order old/fixed matrices; restore fixed source."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
fixed = header.read_bytes()
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root), TMPDIR=str(root / "temp"),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
with (dest / "pytest_final.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest",
                    "-q", "-p", "no:cacheprovider", "-p", "jitpaths", "tests/attention/test_mla_host_batch.py"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("full pytest passed", flush=True)


def normal_compiler(label):
    library = next(path for path in Path("temp/jit_a100_final").rglob("*.so")
                   if path.name.endswith("_host_overlap_v2.so") and "dtype_q_bf16_" in path.name)
    for kind, flag in (("resources", "--dump-resource-usage"), ("sass", "--dump-sass")):
        with (dest / f"{label}_normal_{kind}.txt").open("w") as output:
            subprocess.run(["/usr/local/cuda-13.1/bin/cuobjdump", flag, str(library)],
                           stdout=output, check=True)


normal_compiler("fixed")
try:
    header.write_bytes(subprocess.check_output(["git", "show", "109b1c76:" + str(header)]))
    subprocess.run(["python3", str(dest / "screen.py"), "old_full", "--full"], check=True)
    normal_compiler("old")
    header.write_bytes(fixed)
    subprocess.run(["python3", str(dest / "screen.py"), "fixed_repeat"], check=True)
    header.write_bytes(subprocess.check_output(["git", "show", "5c44fd70:" + str(header)]))
    subprocess.run(["python3", str(dest / "screen.py"), "regressed_repeat"], check=True)
finally:
    header.write_bytes(fixed)
print("verification complete; fixed header restored", flush=True)
