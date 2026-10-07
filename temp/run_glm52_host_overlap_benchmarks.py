# Purpose: Compare native and fused mixed-tier MLA over all requested cases.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_overlap_benchmarks.py
import os
from pathlib import Path
import subprocess
import sys

repository = Path(__file__).resolve().parents[1]
environment = dict(os.environ)
environment.update(
    PYTHONPATH=str(repository),
    FLASHINFER_WORKSPACE_BASE=str(repository / "temp/jit_overlap_host_verified"),
    FLASHINFER_CUDA_ARCH_LIST="8.0",
    TMPDIR=str(repository / "temp"),
    OMP_NUM_THREADS="8",
    MKL_NUM_THREADS="8",
)
for layout in ("random", "head", "tail"):
    for mode in ("native", "overlap"):
        output = repository / f"temp/glm52_host_{mode}_final_{layout}.json"
        if output.exists():
            raise FileExistsError(output)
        command = [
            sys.executable,
            "temp/bench_glm52_sparse_decode_hbm_host_direct.py",
            "--device",
            "1",
            "--layout",
            layout,
            "--rounds",
            "40",
            "--index-sets",
            "8",
            "--output",
            str(output),
        ]
        if mode == "overlap":
            command.append("--host-prefetch")
        print(f"Benchmarking {mode}, layout={layout}", flush=True)
        subprocess.run(command, env=environment, cwd=repository, check=True)
