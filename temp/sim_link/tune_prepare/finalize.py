"""Wait for the active verifier, then measure floors once and generate the final report."""
from pathlib import Path
import subprocess
import time

dest = Path(__file__).resolve().parent
while "verification finished; final restored" not in (dest / "verify.log").read_text():
    log = (dest / "verify.log").read_text()
    if "Traceback" in log:
        raise RuntimeError(log[-4000:])
    time.sleep(20)
subprocess.run(["python3", str(dest / "floors.py")], check=True)
subprocess.run(["python3", str(dest / "report.py")], check=True)
print("floors and report complete", flush=True)
