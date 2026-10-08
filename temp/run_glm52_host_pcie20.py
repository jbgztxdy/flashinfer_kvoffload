# Purpose: Measure the selected Host pipeline against preserved baselines.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_pcie20.py
import os
from pathlib import Path
import subprocess
import sys


def main():
    repository = Path(__file__).resolve().parents[1]
    folder = repository / "temp/host_attention_pcie20"
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(repository),
        FLASHINFER_WORKSPACE_BASE=str(repository / "temp/jit_host_pcie_final"),
        FLASHINFER_CUDA_ARCH_LIST="8.0",
        TMPDIR=str(repository / "temp"),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        MAX_JOBS="8",
    )
    for layout in ("random", "head", "tail"):
        subprocess.run(
            [
                sys.executable,
                "temp/bench_glm52_host_comparison.py",
                "--host-kernel",
                "auto",
                "--layout",
                layout,
                "--rounds",
                "40",
                "--index-sets",
                "8",
                "--baseline-module",
                str(folder / "baseline_v44.so"),
                "--output",
                str(folder / f"after_{layout}.json"),
            ],
            env=environment,
            cwd=repository,
            check=True,
        )


if __name__ == "__main__":
    main()
