"""Verify pytest uses the installed header after the repository JIT path plugin."""
import hashlib
from pathlib import Path
from flashinfer.jit import env
root=Path(__file__).resolve().parents[3]
assert (root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()==(Path(__file__).resolve().parent/'cached_pe.cuh').read_bytes()
env.FLASHINFER_INCLUDE_DIR=root/'include'
