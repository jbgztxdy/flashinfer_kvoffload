# Purpose: Measure paired previous/current Host attention across layouts and contexts.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_25_30.py
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=repository / "temp/host_attention_25_30"
    )
    parser.add_argument("--workspace", default="temp/jit_host_25_30_v42")
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(repository),
        FLASHINFER_WORKSPACE_BASE=str(repository / args.workspace),
        FLASHINFER_CUDA_ARCH_LIST="8.0",
        TMPDIR=str(repository / "temp"),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        MAX_JOBS="8",
    )
    baseline = repository / "temp/host_attention_25_30/baseline_v35.so"
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    for layout in ("random", "head", "tail"):
        command = [
            sys.executable,
            "temp/bench_glm52_host_comparison.py",
            "--layout",
            layout,
            "--rounds",
            "40",
            "--index-sets",
            "8",
            "--baseline-module",
            str(baseline),
            "--output",
            str(args.output_dir / f"after_{layout}.json"),
        ]
        print(f"Paired benchmark: {layout}", flush=True)
        subprocess.run(command, env=environment, cwd=repository, check=True)


if __name__ == "__main__":
    main()
