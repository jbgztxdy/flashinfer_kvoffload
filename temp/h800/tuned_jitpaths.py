"""pytest JIT paths for the separate tuned source snapshot."""
from pathlib import Path

import flashinfer.jit.env as jit_env

root = Path(__file__).resolve().parents[2]
jit_env.FLASHINFER_DATA = root
jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
jit_env.FLASHINFER_INCLUDE_DIR = root / "include"
jit_env.CUTLASS_INCLUDE_DIRS = [root / "3rdparty/cutlass/include", root / "3rdparty/cutlass/tools/util/include"]
jit_env.SPDLOG_INCLUDE_DIR = root / "3rdparty/spdlog/include"
