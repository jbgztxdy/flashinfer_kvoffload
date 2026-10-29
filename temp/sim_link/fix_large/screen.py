"""Same controls for the large-batch regression and small-batch benefit checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
cases = [("proxy_g100_mtp8", "peer", 8, [1, 4, 16, 128], "100:20000", None),
         ("proxy_g100_mtp1", "peer", 1, [1, 4, 16], "100:20000", None)]
if args.full:
    cases += [(f"proxy_{tier}_mtp{mtp}", tier, mtp, [1, 4, 16, 128], None, None)
              for tier in ("peer", "host") for mtp in (1, 8)]
    cases += [(f"proxy_g50_mtp{mtp}", "peer", mtp, [1, 4, 16, 128], "50:20000", None)
              for mtp in (1, 8)]
    cases += [(f"assistant_g{bw}_mtp{mtp}", "peer", mtp, [1, 4, 16], f"{bw}:20000",
               "temp/glm_trace/assistant64k/doc*.pt") for bw in (50, 100) for mtp in (1, 8)]
    cases += [(f"assistant_{tier}_mtp{mtp}", tier, mtp, [1, 4, 16], None,
               "temp/glm_trace/assistant64k/doc*.pt") for tier in ("peer", "host") for mtp in (1, 8)]
for name, tier, mtp, requests, emu, capture in cases:
    output = dest / f"{args.label}_{name}.json"
    assert not output.exists(), output
    run_env = dict(env, FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
    run_env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
               "temp/sim_link/bench_tune.py", "--tier", tier, "--mtp", str(mtp),
               "--requests", *map(str, requests), "--passes", "7", "--fetch-ctas", "16",
               "--host-item-entries", "64", "--modes", "ours_stage", "ours_inplace",
               "--output", str(output)]
    if emu:
        run_env.update(FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
                       FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
        command += ["--emu", emu]
    if capture:
        command += ["--captures", capture, "--steps", str(1024 // mtp)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(output.read_text())
    assert len(data["results"]) == len(requests)
    assert all(error < .01 for row in data["results"] for error in row["correctness_max_abs_error"].values())
    print("verified", output.name, flush=True)
