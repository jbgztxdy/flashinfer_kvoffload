"""Print measured completion and the active matrix group without reading raw samples."""
from pathlib import Path
import json
import subprocess
import time

DEST=Path(__file__).resolve().parent
counts={"floor":0,"snapshots":0,"nocache":0,"lru":0}
copy_pending=0
for path in (DEST/"raw").glob("*.json"):
    if path.name.startswith("smoke_"):continue
    try:data=json.loads(path.read_text())
    except FileNotFoundError:continue
    if data["complete"]:
        if data["configuration"]["kind"]=="nocache" and data["configuration"].get("memcpy_path_version",1)<2:
            copy_pending+=len(data["cases"]);continue
        counts[data["configuration"]["kind"]]+=len(data["cases"])
logs=sorted(DEST.glob("run*.log"),key=lambda p:p.stat().st_mtime)
print(json.dumps(dict(timestamp=time.strftime("%FT%T%z"),counts=counts,copy_baseline_remeasurement_pending=copy_pending,
    expected=dict(floor=39,snapshots=273,nocache=4641,lru=663),
    latest_log=logs[-1].name if logs else None,
    last_lines=logs[-1].read_text().splitlines()[-4:] if logs else []),ensure_ascii=False))
