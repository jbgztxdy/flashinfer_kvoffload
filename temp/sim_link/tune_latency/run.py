"""Serial same-parameter latency experiments; do not change the header during a run."""
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
parser.add_argument("--suite", choices=("screen", "full"), default="screen")
parser.add_argument("--case", nargs="+", default=None)
parser.add_argument("--fetch", type=int, default=16)
parser.add_argument("--items", type=int, default=64)
args = parser.parse_args()
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
cases = [(f"proxy_g100_l{lat}_mtp{mtp}", "peer", mtp, f"100:{lat}", None)
         for lat in (0, 20000) for mtp in (1, 8)]
cases += [(f"proxy_g50_l20000_mtp{mtp}", "peer", mtp, "50:20000", None) for mtp in (1, 8)]
cases += [(f"proxy_{tier}_mtp{mtp}", tier, mtp, None, None)
          for tier in ("host", "peer") for mtp in (1, 8)]
if args.suite == "full":
    cases += [(f"assistant_g{bw}_l20000_mtp{mtp}", "peer", mtp, f"{bw}:20000",
               "temp/glm_trace/assistant64k/doc*.pt") for bw in (50, 100) for mtp in (1, 8)]
    cases += [(f"assistant_{tier}_mtp{mtp}", tier, mtp, None,
               "temp/glm_trace/assistant64k/doc*.pt") for tier in ("peer", "host") for mtp in (1, 8)]
if args.case:
    assert set(args.case) <= {case[0] for case in cases}
    cases = [case for case in cases if case[0] in args.case]
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
header_sha = hashlib.sha256(header.read_bytes()).hexdigest()
for name, tier, mtp, emu, capture in cases:
    output = dest / f"{args.label}_{name}.json"
    assert not output.exists(), output
    run_env = dict(env, FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
    run_env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
               "temp/sim_link/bench_tune.py", "--tier", tier, "--mtp", str(mtp),
               "--requests", "1", "4", "16", "--passes", str(args.passes),
               "--fetch-ctas", str(args.fetch), "--host-item-entries", str(args.items),
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
    assert data["source_sha256"][str(header)] == header_sha
    assert len(data["results"]) == 3
    assert all(error < .01 for row in data["results"] for error in row["correctness_max_abs_error"].values())
    print("verified", output.name, flush=True)
