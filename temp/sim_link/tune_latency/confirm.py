"""Run the exact old kernel after the candidate; always restore the candidate header."""
from pathlib import Path
import os
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
candidate = header.read_bytes()
baseline = subprocess.check_output(["git", "show", "109b1c76:" + str(header)])
try:
    header.write_bytes(baseline)
    subprocess.run(["python3", str(dest / "run.py"), "base", "--suite", "full"], check=True)
finally:
    header.write_bytes(candidate)
print("full baseline complete; candidate restored", flush=True)
