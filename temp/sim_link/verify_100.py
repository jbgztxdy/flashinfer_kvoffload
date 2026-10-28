"""Serial A/B validation on idle local A100 GPU0/1; restore the candidate header on all exits.

The old header is read from the recorded baseline commit. No other source file is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path("temp/sim_link/tune_100"))
DEST = parser.parse_args().output_dir
DEST.mkdir(parents=True, exist_ok=True)
HEADER = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = HEADER.read_bytes()
baseline = subprocess.check_output(["git", "show", "c7921958:" + str(HEADER)])
assert hashlib.sha256(baseline).hexdigest() == "9ef5c30414ef383452d38719c7114063626b0f988349cd67ec53bf3477f1b884"
assert b"params.fetch_ctas <= 8 ? 3 : 1" in candidate
env_base = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
                CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
                PYTHONPATH=str(ROOT), TMPDIR=str(ROOT / "temp"))
env_base.pop("FLASHINFER_HOST_BATCH_GRID_LIMIT", None)
records = []


def run(name, mtp, requests, latency=20000, fetch=16, item=64, modes=None, normal=False, bandwidth=100):
    output = DEST / (name + ".json")
    if output.exists():
        raise FileExistsError(output)
    env = dict(env_base, FLASHINFER_WORKSPACE_BASE=str(ROOT / "temp/jit_emu"))
    env["FLASHINFER_EXTRA_CUDAFLAGS"] = "-DFLASHINFER_FAR_EMU"
    if normal:
        env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
        env.pop("FLASHINFER_FAR_EMU", None)
        env["FLASHINFER_WORKSPACE_BASE"] = str(ROOT / "temp/jit_a100_final")
    cmd = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
           "temp/sim_link/bench_tune.py", "--tier", "peer", "--mtp", str(mtp),
           "--requests", *map(str, requests), "--passes", "7", "--fetch-ctas", str(fetch),
           "--host-item-entries", str(item), "--modes", *(modes or ["ours_stage", "ours_inplace"]),
           "--output", str(output)]
    if not normal:
        cmd += ["--emu", f"{bandwidth}:{latency}"]
    with (DEST / (name + ".log")).open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.read_text())
    assert len(result["results"]) == len(requests)
    assert all(e < .01 for r in result["results"] for e in r["correctness_max_abs_error"].values())
    records.append(dict(name=name, command=cmd, source_sha256=result["source_sha256"]))
    (DEST / "verification_runs.json").write_text(json.dumps(records, indent=2) + "\n")
    print("verified", name, flush=True)


try:
    HEADER.write_bytes(baseline)
    for latency in (0, 20000):
        run(f"verify_base_mtp8_l{latency}", 8, [1, 2, 4, 16, 64, 128], latency=latency,
            modes=["ours_stage", "memcpy_lru", "direct_lru"])
    run("verify_base_mtp1_l20000", 1, [1, 4, 16, 128],
        modes=["ours_stage", "memcpy_lru", "direct_lru"])
    HEADER.write_bytes(candidate)
    for latency in (0, 20000):
        run(f"verify_final_mtp8_l{latency}", 8, [1, 2, 4, 16, 64, 128], latency=latency)
    run("verify_final_mtp1_l20000", 1, [1, 4, 16, 128])
    run("verify_final_mtp8_l20000_f24", 8, [1, 2, 4, 16, 64, 128], fetch=24)
    run("verify_final_mtp8_l20000_h256", 8, [1, 2, 4], item=256)
    run("verify_final_mtp8_l0_h256", 8, [1, 2, 4], latency=0, item=256)
    run("verify_final_mtp1_l20000_f32", 1, [1, 4, 16, 128], fetch=32)
    run("floor_mtp8", 8, [1, 2, 4, 16, 64, 128], normal=True, modes=["floor_fa2"])
    run("floor_mtp1", 1, [1, 4, 16, 128], normal=True, modes=["floor_fa2"])
    run("verify_final_mtp8_g50_l20000", 8, [1, 4, 16, 64, 128], bandwidth=50)
finally:
    HEADER.write_bytes(candidate)
    assert HEADER.read_bytes() == candidate
print("A/B complete; candidate header restored", hashlib.sha256(candidate).hexdigest(), flush=True)
