"""Serial LRU experiment suite on local A100 GPU0/1. Labels must have fresh outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--suite", choices=("screen", "full"), default="screen")
ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--output-dir", type=Path, default=Path("temp/sim_link/tune_lru"))
args = ap.parse_args()
dest = args.output_dir
dest.mkdir(parents=True, exist_ok=True)
source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, (
    "include/flashinfer/attention/mla_host_batch.cuh",
    "include/flashinfer/attention/far_link_emu.cuh", "temp/sim_link/bench_tune.py", __file__))}
(dest / (args.label + "_source.cuh")).write_bytes(Path("include/flashinfer/attention/mla_host_batch.cuh").read_bytes())
env_base = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
                CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
                PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
env_base.pop("FLASHINFER_HOST_BATCH_GRID_LIMIT", None)
requests8 = [1, 2, 4, 16] if args.suite == "screen" else [1, 2, 4, 16, 64, 128]
requests1 = [1, 4, 16] if args.suite == "screen" else [1, 4, 16, 128]
cases = [(f"mtp{mtp}_l{lat}", mtp, reqs, lat, "peer", None, 65536, None)
         for lat in (0, 20000) for mtp, reqs in ((8, requests8), (1, requests1))]
if args.suite == "full":
    cases += [("host_mtp8", 8, [1, 4, 16, 64], None, "host", None, 65536, None),
              ("glm8k_mtp1", 1, [1, 4], None, "peer", "temp/glm_trace/validation8k/doc0.pt", 8192, 256),
              ("g50_mtp8_l20000", 8, requests8, 20000, "peer", None, 65536, None)]
for case, mtp, reqs, lat, tier, captures, context, steps in cases:
    output = dest / f"{args.label}_{case}.json"
    if output.exists():
        raise FileExistsError(output)
    env = dict(env_base, FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
    env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
    env.pop("FLASHINFER_FAR_EMU", None)
    cmd = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
           "temp/sim_link/bench_tune.py", "--tier", tier, "--mtp", str(mtp),
           "--requests", *map(str, reqs), "--passes", str(args.passes), "--fetch-ctas", "16",
           "--modes", "ours_stage", "ours_inplace", "--context", str(context), "--output", str(output)]
    if lat is not None:
        env["FLASHINFER_EXTRA_CUDAFLAGS"] = "-DFLASHINFER_FAR_EMU"
        env["FLASHINFER_WORKSPACE_BASE"] = str(root / "temp/jit_emu")
        cmd += ["--emu", f"{50 if case.startswith('g50') else 100}:{lat}"]
    if captures:
        cmd += ["--captures", captures]
    if steps:
        cmd += ["--steps", str(steps)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(output.read_text())
    assert data["source_sha256"]["include/flashinfer/attention/mla_host_batch.cuh"] == source["include/flashinfer/attention/mla_host_batch.cuh"]
    assert len(data["results"]) == len(reqs)
    assert all(e < .01 for r in data["results"] for e in r["correctness_max_abs_error"].values())
    print("verified", output.name, flush=True)
(dest / (args.label + "_manifest.json")).write_text(json.dumps(dict(
    label=args.label, suite=args.suite, passes=args.passes, source_sha256=source,
    files=[str(dest / f"{args.label}_{case[0]}.json") for case in cases]), indent=2) + "\n")
