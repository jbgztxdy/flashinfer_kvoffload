import os
from pathlib import Path
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR=Path(__file__).resolve().parent/('include_'+os.environ['TUNE_VARIANT'])
