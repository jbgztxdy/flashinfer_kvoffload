"""Screen planner settings on unchanged real assistant top-k, on A100 GPU0/1."""
import argparse
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("--mtp", type=int, default=8)
parser.add_argument("--bandwidth", type=int, default=100)
args = parser.parse_args()
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
           FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
for fetch in (16, 32):
    for items, split in ((64, 1), (256, 1), (1024, 1), (64, 0)):
        output = dest / f"screen_g{args.bandwidth}_mtp{args.mtp}_f{fetch}_h{items}_s{split}.json"
        assert not output.exists()
        command = ["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                   "temp/sim_link/bench_tune.py", "--tier", "peer", "--mtp", str(args.mtp),
                   "--requests", "1", "4", "16", "--passes", "5", "--fetch-ctas", str(fetch),
                   "--host-item-entries", str(items), "--host-split", str(split),
                   "--captures", "temp/glm_trace/assistant64k/doc*.pt", "--steps", str(256 // args.mtp),
                   "--emu", f"{args.bandwidth}:20000", "--modes", "ours_stage", "--output", str(output)]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("verified", output.name, flush=True)
