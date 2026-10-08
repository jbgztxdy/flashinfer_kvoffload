# Purpose: Run matched Host-attention comparisons for random, head, and tail hits.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_comparison.py
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
folder = root / "temp/host_attention_comparison"
folder.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ)
environment.update(
    PYTHONPATH=str(root),
    FLASHINFER_WORKSPACE_BASE=str(root / "temp/jit_overlap_host_verified"),
    FLASHINFER_CUDA_ARCH_LIST="8.0",
    TMPDIR=str(root / "temp"),
    OMP_NUM_THREADS="8",
    MKL_NUM_THREADS="8",
)
for layout in ("random", "head", "tail"):
    output = folder / f"raw_{layout}.json"
    if output.exists():
        raise FileExistsError(output)
    print(f"Benchmarking {layout}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "temp/bench_glm52_host_comparison.py",
            "--layout",
            layout,
            "--output",
            str(output),
        ],
        env=environment,
        cwd=root,
        check=True,
    )
