"""Validate the single-query candidate, then repeat the old version; always restore it."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
baseline = subprocess.check_output(["git", "show", "aeca4403:" + str(header)])
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           FLASHINFER_CUDA_ARCH_LIST="8.0", FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"),
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root), TMPDIR=str(root / "temp"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
with (dest / "pytest_final.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest",
                    "-q", "-p", "no:cacheprovider", "-p", "jitpaths",
                    "tests/attention/test_mla_host_batch.py"], env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)
print("full pytest passed", flush=True)
glm_args = ["--glm", "--case", "glm64k_peer_mtp1", "glm64k_host_mtp1",
            "glm64k_g100_l20000_mtp1", "--requests", "1", "4"]


def run(label, extra=()):
    subprocess.run(["python3", str(dest / "run.py"), label, *extra], check=True)


def large(label):
    for case, emu in (("g100_l20000_mtp8_q128", "100:20000"), ("peer_mtp8_q128", None)):
        out = dest / f"{label}_{case}.json"
        assert not out.exists()
        run_env = dict(env, PYTHONPATH=str(root))
        command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                   "temp/sim_link/bench_tune.py", "--tier", "peer", "--mtp", "8",
                   "--requests", "128", "--passes", "7", "--fetch-ctas", "16",
                   "--host-item-entries", "64", "--modes", "ours_stage", "ours_inplace",
                   "--output", str(out)]
        if emu:
            run_env.update(FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
                           FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
            command += ["--emu", emu]
        with out.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("verified", out.name, flush=True)


try:
    run("single_query", glm_args)
    large("single_query")
    header.write_bytes(baseline)
    run("base_repeat")
    run("base_repeat", glm_args)
    large("base_repeat")
finally:
    header.write_bytes(candidate)
print("confirmation complete; single-query candidate restored", flush=True)
