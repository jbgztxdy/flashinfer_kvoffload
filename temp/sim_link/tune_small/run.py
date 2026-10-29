"""Serial same-parameter phase/merge experiments on A100 GPU0/1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--passes", type=int, default=7)
parser.add_argument("--glm", action="store_true")
parser.add_argument("--case", nargs="+", default=None)
parser.add_argument("--requests", nargs="+", type=int, default=None)
args = parser.parse_args()
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
cases = [(f"g100_l{lat}_mtp{mtp}", "peer", mtp, f"100:{lat}", None)
         for lat in (0, 20000) for mtp in (1, 8)]
cases += [(f"{tier}_mtp{mtp}", tier, mtp, None, None)
          for tier in ("peer", "host") for mtp in (1, 8)]
if args.glm:
    cases = [(f"glm64k_{name}", tier, mtp, emu, "temp/glm_trace/replay64k/doc0.pt")
             for name, tier, mtp, emu, _ in cases]
if args.case:
    assert set(args.case) <= {case[0] for case in cases}
    cases = [case for case in cases if case[0] in args.case]
requests = args.requests or [1, 4, 16]
for name, tier, mtp, emu, capture in cases:
    output = dest / f"{args.label}_{name}.json"
    assert not output.exists(), output
    run_env = dict(env, FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
    run_env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
               "temp/sim_link/bench_tune.py", "--tier", tier, "--mtp", str(mtp),
               "--requests", *map(str, requests), "--passes", str(args.passes),
               "--fetch-ctas", "16", "--host-item-entries", "64",
               "--modes", "ours_stage", "ours_inplace", "--output", str(output)]
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
