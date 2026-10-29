"""Build a H800-only header from the existing cluster baseline and measured tuning patches.

The source tree is a separate detached worktree, so pending production jobs keep their baseline.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

root = Path(__file__).resolve().parents[2]
header = "include/flashinfer/attention/mla_host_batch.cuh"
patches = []
for name, revisions in (("fetch", ["c7921958", "d92e7e83"]),
                        ("lru", ["d92e7e83", "614655f1"]),
                        ("single_query", ["aeca4403"])):
    patch = f"temp/h800/tuned_{name}.patch"
    (root / patch).write_bytes(subprocess.check_output(
        ["git", "diff", *revisions, "--", header], cwd=root))
    patches.append(patch)
with tempfile.TemporaryDirectory(prefix="fi_h800_tuned_") as temporary:
    tree = Path(temporary)
    dest = tree / header
    dest.parent.mkdir(parents=True)
    baseline = subprocess.check_output(["git", "show", "dbbc02c6:" + header], cwd=root)
    dest.write_bytes(baseline)
    for patch in patches:
        subprocess.run(["patch", "--batch", "-p1", "-i", str(root / patch)], cwd=tree, check=True)
    assert b"FAR_EMU" not in dest.read_bytes()
    # Derive the same H800 code independently by removing only emulator hooks from the local header.
    current = (root / header).read_text().splitlines(keepends=True)
    stripped, in_hook = [], False
    for line in current:
        if '#include "far_link_emu.cuh"' in line:
            continue
        if "FAR_EMU_" in line:
            in_hook = ";" not in line
            continue
        if in_hook:
            in_hook = ";" not in line
            continue
        stripped.append(line)
    # Zero-context patches may carry harmless blank lines; compare all nonempty lines exactly.
    assert [line for line in dest.read_text().splitlines() if line.strip()] == [
        line for line in "".join(stripped).splitlines() if line.strip()]
    files = [header, "tests/attention/test_mla_host_batch.py", "temp/h800/bench_rounds.py",
             "temp/h800/bench_sweep.py", "temp/h800/tiers.py", "temp/h800/tuned_assistant.sbatch",
             "temp/h800/tuned_jitpaths.py"]
    for name in files[1:]:
        output = tree / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((root / name).read_bytes())
    meta = dict(cluster_base="dbbc02c6", local_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        local_kernel_sha256=hashlib.sha256((root / header).read_bytes()).hexdigest(),
        emulator_hooks=False, independent_derivations_match=True,
        patches={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in patches},
        files={name: hashlib.sha256((tree / name).read_bytes()).hexdigest() for name in files})
    (tree / "SOURCE_MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    archive = root / "temp/h800/tuned_20261029.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in files + ["SOURCE_MANIFEST.json"]:
            bundle.add(tree / name, arcname=name)
    (root / "temp/h800/tuned_20261029_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(archive)
