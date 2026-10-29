"""Finish the optional pipeline smoke and audit after the default-path verifier releases GPU3."""
from pathlib import Path
import subprocess
import time

dest = Path(__file__).resolve().parent
while "accepted verification finished; source restored" not in (dest / "accept.log").read_text():
    log = (dest / "accept.log").read_text()
    if "Traceback" in log:
        raise RuntimeError(log[-4000:])
    time.sleep(20)
subprocess.run(["python3", str(dest / "screen.py"), "optin", "--compact-fetch"], check=True)
subprocess.run(["python3", str(dest / "host_recheck.py")], check=True)
subprocess.run(["python3", str(dest / "report.py")], check=True)
subprocess.run(["python3", str(dest / "report.py"), "--label", "final"], check=True)
subprocess.run(["python3", str(dest / "audit.py")], check=True)
print("accepted report and audit complete", flush=True)
