import os
from pathlib import Path
from flashinfer.jit import env
p=Path(__file__).resolve().parent
env.FLASHINFER_INCLUDE_DIR=p/'include_v2'
