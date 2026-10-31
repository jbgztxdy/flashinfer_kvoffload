"""Pytest uses the frozen header that is byte-identical to the production header."""
from pathlib import Path
from flashinfer.jit import env
p=Path(__file__).resolve().parent
env.FLASHINFER_INCLUDE_DIR=p/'include_final'
