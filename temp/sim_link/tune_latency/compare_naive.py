"""Compare the final staged kernel with both naive methods on complete assistant windows."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0",
           PYTHONPATH=str(root), TMPDIR=str(root / "temp"),
           FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_emu"),
           FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
for bandwidth, mtp in ((100, 1), (100, 8), (50, 8)):
    output = dest / f"comparison_assistant_g{bandwidth}_mtp{mtp}.json"
    assert not output.exists()
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                        "temp/sim_link/bench_tune.py", "--tier", "peer", "--passes", "7",
                        "--fetch-ctas", "16", "--host-item-entries", "64", "--output", str(output),
                        "--mtp", str(mtp), "--requests", "1", "4", "16", "--steps", str(1024 // mtp),
                        "--captures", "temp/glm_trace/assistant64k/doc*.pt",
                        "--modes", "ours_stage", "memcpy_lru", "direct_lru", "floor_fa2",
                        "--emu", f"{bandwidth}:20000"], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print("verified", output.name, flush=True)
