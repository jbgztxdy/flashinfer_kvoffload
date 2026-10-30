"""Refresh the partial report and audit once the full matrix finishes."""
from pathlib import Path
import subprocess
import time

DEST=Path(__file__).resolve().parent
while True:
    subprocess.run(["python3",str(DEST/"report.py")],check=True)
    subprocess.run(["python3",str(DEST/"plot.py")],check=True)
    logs=[p.read_text() for p in DEST.glob("run*.log")]
    if any("ALL_MATRIX_COMPLETE" in s for s in logs):
        audit=subprocess.run(["python3",str(DEST/"audit.py")])
        if audit.returncode:
            time.sleep(30)
            continue
        subprocess.run(["python3",str(DEST/"report.py")],check=True)
        subprocess.run(["python3",str(DEST/"export_html.py")],check=True)
        print("REPORT_AND_AUDIT_COMPLETE",flush=True)
        break
    time.sleep(60)
