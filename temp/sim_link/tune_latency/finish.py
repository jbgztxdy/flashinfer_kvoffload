"""Final correctness, missing native floor and large-batch non-regression checks."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root / "temp/pytest_shim") + ":" + str(root), TMPDIR=str(root / "temp"),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
with (dest / "pytest_final.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python", "-m", "pytest",
                    "-q", "-p", "no:cacheprovider", "-p", "jitpaths", "tests/attention/test_mla_host_batch.py"],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("91 tests passed", flush=True)


def bench(output, extra, emu=None):
    assert not output.exists()
    run_env = dict(env, PYTHONPATH=str(root))
    if emu:
        run_env.update(FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
                       FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
        extra = [*extra, "--emu", emu]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                        "temp/sim_link/bench_tune.py", "--tier", "peer", "--passes", "7",
                        "--fetch-ctas", "16", "--host-item-entries", "64", "--output", str(output),
                        *extra], env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("verified", output.name, flush=True)


bench(dest / "floor_assistant_mtp1.json", ["--mtp", "1", "--requests", "1", "4", "16",
      "--captures", "temp/glm_trace/assistant64k/doc*.pt", "--steps", "1024", "--modes", "floor_fa2"])
baseline = subprocess.check_output(["git", "show", "109b1c76:" + str(header)])
try:
    for label, source in (("base", baseline), ("final", candidate)):
        header.write_bytes(source)
        for case, emu in (("g100", "100:20000"), ("g50", "50:20000"), ("peer", None)):
            bench(dest / f"{label}_large_{case}_mtp8.json",
                  ["--mtp", "8", "--requests", "128", "--modes", "ours_stage", "ours_inplace"], emu)
        for kind, capture in (("proxy", None), ("assistant", "temp/glm_trace/assistant64k/doc*.pt")):
            extra = ["--mtp", "1", "--requests", "1", "4", "16", "--fetch-ctas", "32",
                     "--modes", "ours_stage", "ours_inplace"]
            if capture:
                extra += ["--captures", capture, "--steps", "1024"]
            bench(dest / f"{label}_tuned_{kind}_g100_mtp1.json", extra, "100:20000")
    for case in ("proxy_g100_l20000_mtp1", "proxy_g100_l20000_mtp8"):
        subprocess.run(["python3", str(dest / "run.py"), "final_repeat", "--case", case], check=True)
    header.write_bytes(baseline)
    for case in ("proxy_g100_l20000_mtp1", "proxy_g100_l20000_mtp8"):
        subprocess.run(["python3", str(dest / "run.py"), "base_repeat", "--case", case], check=True)
finally:
    header.write_bytes(candidate)
print("final checks complete; final header restored", flush=True)
