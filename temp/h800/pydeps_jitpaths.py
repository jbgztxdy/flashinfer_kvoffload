# pytest plugin (-p jitpaths): point FlashInfer's JIT at the source checkout, as temp/bench_* do.
import pathlib
import flashinfer.jit.env as jit_env
_r = pathlib.Path(__file__).resolve().parents[3]
jit_env.FLASHINFER_DATA = _r
jit_env.FLASHINFER_CSRC_DIR = _r / "csrc"
jit_env.FLASHINFER_INCLUDE_DIR = _r / "include"
jit_env.CUTLASS_INCLUDE_DIRS = [_r / "3rdparty/cutlass/include", _r / "3rdparty/cutlass/tools/util/include"]
jit_env.SPDLOG_INCLUDE_DIR = _r / "3rdparty/spdlog/include"
