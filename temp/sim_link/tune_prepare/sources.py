"""Restore untracked source snapshots from committed baselines and evidence patches."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
header = "include/flashinfer/attention/mla_host_batch.cuh"
base = subprocess.check_output(["git", "show", "f78ac492:" + header], cwd=root)
(dest / "base.cuh").write_bytes(base)
for label in ("final", "accepted", "retained", "guarded"):
    with tempfile.TemporaryDirectory(dir=root / "temp", prefix="prepare_sources_") as directory:
        for name in (header, "tests/attention/test_mla_host_batch.py", "temp/h800/tiers.py"):
            target = Path(directory) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", "ac507996:" + name], cwd=root))
        subprocess.run(["patch", "--silent", "-p1", "-i", str(dest / f"{label}.patch")], cwd=directory, check=True)
        (dest / f"{label}.cuh").write_bytes((Path(directory) / header).read_bytes())
print("source snapshots reconstructed from git + patches")
