"""Controlled tuning on the free A100 PCIe: separate HBM allocations + unchanged link emulator."""
import argparse
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--full", action="store_true")
parser.add_argument("--compact-fetch", action="store_true", help="experimental high-latency fetch ring")
parser.add_argument("--small-prepare", action="store_true", help="enable the optional small-prepare optimization")
args = parser.parse_args()
cases = [(f"proxy_g100_mtp{mtp}", mtp, [1, 4, 16, 128] if args.full else [1, 4, 16], "100:20000", False)
         for mtp in (1, 8)]
if args.full:
    cases += [(f"proxy_g50_mtp{mtp}", mtp, [1, 4, 16, 128], "50:20000", False) for mtp in (1, 8)]
    cases += [(f"assistant_g{bw}_mtp{mtp}", mtp, [1, 4, 16], f"{bw}:20000", True)
              for bw in (50, 100) for mtp in (1, 8)]
    cases += [(f"proxy_g100_l0_mtp{mtp}", mtp, [1, 4, 16, 128], "100:0", False) for mtp in (1, 8)]
    cases += [(f"proxy_host_mtp{mtp}", mtp, [1, 4, 16, 128], None, False) for mtp in (1, 8)]
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(root),
           TMPDIR=str(root / "temp"), FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU",
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu_prepare"))
if args.compact_fetch:
    if "FLASHINFER_HOST_BATCH_COMPACT_FETCH" not in (root / "include/flashinfer/attention/mla_host_batch.cuh").read_text():
        raise SystemExit("Compact fetch is only in the saved accepted.patch experiment; restore that source first.")
    env["FLASHINFER_EXTRA_CUDAFLAGS"] += " -DFLASHINFER_HOST_BATCH_COMPACT_FETCH"
    env["FLASHINFER_WORKSPACE_BASE"] = str(root / "temp/jit_emu_prepare_optin")
if args.small_prepare:
    env["FLASHINFER_EXTRA_CUDAFLAGS"] += " -DFLASHINFER_HOST_BATCH_SMALL_PREPARE"
    env["FLASHINFER_WORKSPACE_BASE"] = str(root / "temp/jit_emu_prepare_guarded")
for name, mtp, requests, emu, assistant in cases:
    output = dest / f"{args.label}_{name}.json"
    assert not output.exists(), output
    run_env = dict(env)
    if emu is None:
        run_env.pop("FLASHINFER_EXTRA_CUDAFLAGS")
        run_env["FLASHINFER_WORKSPACE_BASE"] = str(root / "temp/jit_prepare_normal")
        if args.small_prepare:
            run_env["FLASHINFER_EXTRA_CUDAFLAGS"] = "-DFLASHINFER_HOST_BATCH_SMALL_PREPARE"
            run_env["FLASHINFER_WORKSPACE_BASE"] = str(root / "temp/jit_prepare_guarded_normal")
    command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
               "temp/sim_link/bench_tune.py", "--tier", "peer" if emu else "host", "--peer", "0", "--mtp", str(mtp),
               "--requests", *map(str, requests), "--passes", "7", "--fetch-ctas", "16",
               "--host-item-entries", "64", "--modes", "ours_stage", "ours_inplace", "floor_fa2",
               "--output", str(output)]
    if emu:
        command += ["--emu", emu]
    if assistant:
        command += ["--captures", "temp/glm_trace/assistant64k/doc*.pt", "--steps", str(1024 // mtp)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(output.read_text())
    assert all(v < .01 for row in data["results"] for v in row["correctness_max_abs_error"].values())
    print("verified", output.name, flush=True)
