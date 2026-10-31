"""Verify pytest uses the installed header after the repository JIT path plugin."""
import hashlib
from pathlib import Path
from flashinfer.jit import env
root=Path(__file__).resolve().parents[3]
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
env.FLASHINFER_INCLUDE_DIR=root/'include'
