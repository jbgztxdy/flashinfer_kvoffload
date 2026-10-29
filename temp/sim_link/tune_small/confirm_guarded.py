"""Repeat the baseline with bounded enqueue guards; restore the candidate on exit."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
baseline = subprocess.check_output(["git", "show", "aeca4403:" + str(header)])
try:
    header.write_bytes(baseline)
    for extra in (("--case", "g100_l0_mtp1", "g100_l20000_mtp1", "peer_mtp1", "host_mtp1"),
                  ("--glm", "--case", "glm64k_g100_l20000_mtp1", "glm64k_peer_mtp1", "glm64k_host_mtp1")):
        subprocess.run(["python3", str(dest / "run.py"), "guarded_base",
                        *extra, "--requests", "1", "4"], check=True)
finally:
    header.write_bytes(candidate)
print("guarded baseline complete; candidate restored", flush=True)

env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           FLASHINFER_CUDA_ARCH_LIST="8.0", FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_a100_final"),
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
floor = dest / "guarded_floor_glm64k_peer_mtp1.json"
assert not floor.exists()
with floor.with_suffix(".log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                    "temp/h800/bench_rounds.py", "--tier", "peer", "--mtp", "1",
                    "--requests", "1", "4", "--steps", "1024", "--passes", "7",
                    "--captures", "temp/glm_trace/replay64k/doc0.pt", "--modes", "floor_fa2",
                    "--output", str(floor)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("guarded mixed-role floor complete", flush=True)
with (dest / "assistant_replay_run.log").open("w") as log:
    subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                    "temp/glm_trace/run_64k_replay.py", "--captures", "temp/glm_trace/assistant64k/doc*.pt",
                    "--out", "temp/glm_trace/results/a100_64k_assistant_20261029",
                    "--assistant-only"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print("assistant replay complete", flush=True)
