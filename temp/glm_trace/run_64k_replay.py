"""Serial local A100 64K/1024 replay, with native and simulated far links."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import torch

root = Path(__file__).resolve().parents[2]
os.chdir(root)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--captures", default="temp/glm_trace/replay64k/doc0.pt")
parser.add_argument("--out", type=Path, default=Path("temp/glm_trace/results/a100_64k_doc0"))
parser.add_argument("--passes", type=int, default=7)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


traces = []
for path in sorted(map(Path, glob.glob(args.captures))):
    data = torch.load(path, map_location="cpu")
    assert data["tokens"] == 65536 and data["selections"].shape[2:] == (1024, 2048)
    traces.append(dict(path=str(path), sha256=digest(path), layers=data["layers"],
                       group_starts=data["group_starts"].tolist(),
                       group_role_counts=[{name: int((data["role"][pos] == i).sum())
                                           for i, name in enumerate(data["role_names"])}
                                          for pos in data["positions"]],
                       provenance=data["replay_provenance"]))
    del data
assert traces
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
cases = [("peer_mtp1", "peer", 1, [1, 4], None),
         ("peer_mtp8", "peer", 8, [1, 4, 16], None),
         ("host_mtp1", "host", 1, [1, 4], None),
         ("g100_l20_mtp1", "peer", 1, [1, 4], "100:20000"),
         ("g100_l20_mtp8", "peer", 8, [1, 4, 16], "100:20000"),
         ("g50_l20_mtp8", "peer", 8, [1, 4, 16], "50:20000")]
outputs = []
for name, tier, mtp, requests, emu in cases:
    out = args.out / f"{name}.json"
    if out.exists():
        raise FileExistsError(out)
    run_env = dict(env, FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"))
    command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
               "temp/sim_link/bench_tune.py", "--tier", tier, "--mtp", str(mtp),
               "--requests", *map(str, requests), "--passes", str(args.passes),
               "--fetch-ctas", "16", "--host-item-entries", "64",
               "--modes", "ours_stage", "ours_inplace", "memcpy_lru", "direct_lru", "floor_fa2",
               "--context", "65536", "--steps", str(1024 // mtp),
               "--captures", args.captures, "--output", str(out)]
    if emu:
        run_env.update(FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
                       FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
        command += ["--emu", emu]
    with out.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    measured = json.loads(out.read_text())
    assert len(measured["results"]) == len(requests)
    for result in measured["results"]:
        assert all(error < .01 for error in result["correctness_max_abs_error"].values())
        # Same ordering and random seed as tiers.Requests; record the actual source windows.
        combos = [(c, l, g) for c, trace in enumerate(traces)
                  for l in range(len(trace["layers"])) for g in range(len(trace["group_starts"]))]
        random.Random(measured["configuration"]["seed"] + result["requests"]).shuffle(combos)
        bindings = []
        for r in range(result["requests"]):
            c, l, g = combos[r % len(combos)]
            trace = traces[c]
            bindings.append(dict(request=r, capture=trace["path"], layer=trace["layers"][l],
                                 group=g, start=trace["group_starts"][g],
                                 role_counts=trace["group_role_counts"][g]))
        result["replay_assignments"] = bindings
    measured["capture_provenance"] = traces
    out.write_text(json.dumps(measured, indent=1) + "\n")
    outputs.append(out)
    print("verified", out, flush=True)
floors = [p for p in outputs if not json.loads(p.read_text())["emulated_link"]]
subprocess.run(["python3", "temp/glm_trace/06_report_replay.py", *map(str, outputs),
                "--hbm-runs", *map(str, floors), "--out", str(args.out / "results.md")], check=True)
manifest = dict(captures=traces, outputs={str(p): digest(p) for p in outputs},
                source_sha256={str(p): digest(p) for p in map(Path, (
                    __file__, "temp/glm_trace/06_report_replay.py", "temp/sim_link/bench_tune.py",
                    "temp/h800/bench_rounds.py", "temp/h800/tiers.py",
                    "include/flashinfer/attention/mla_host_batch.cuh",
                    "include/flashinfer/attention/far_link_emu.cuh"))})
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
