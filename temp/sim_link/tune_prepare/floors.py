"""Pure-device assistant floors on this A100, compiled without emulation hooks."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="3",
           CUDA_HOME="/usr/local/cuda-13.1", FLASHINFER_CUDA_ARCH_LIST="8.0", TMPDIR=str(root / "temp"),
           PYTHONPATH=str(root), FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_prepare_normal"))
env.pop("FLASHINFER_EXTRA_CUDAFLAGS", None)
env.pop("FLASHINFER_FAR_EMU", None)
for mtp in (1, 8):
    output = dest / f"floor_assistant_mtp{mtp}.json"
    assert not output.exists()
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(["numactl", "-N0", "-m0", "/mnt/data/mhy-vllm-venv/bin/python",
                        "temp/h800/bench_rounds.py", "--tier", "peer", "--peer", "0", "--mtp", str(mtp),
                        "--requests", "1", "4", "16", "--captures", "temp/glm_trace/assistant64k/doc*.pt",
                        "--steps", str(1024 // mtp), "--passes", "7", "--fetch-ctas", "16",
                        "--modes", "floor_fa2", "--output", str(output)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("verified pure HBM", output.name, flush=True)
