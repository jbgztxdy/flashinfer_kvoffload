"""Validate provenance, accepted enqueue segments, references, and evidence coverage."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

dest = Path(__file__).resolve().parent
root = dest.parents[2]
header = "include/flashinfer/attention/mla_host_batch.cuh"


def sha(data):
    return hashlib.sha256(data).hexdigest()


versions = {name: sha(subprocess.check_output(["git", "show", f"{ref}:{header}"], cwd=root))
            for name, ref in (("old", "109b1c76"), ("regressed", "5c44fd70"))}
versions["fixed"] = sha((root / header).read_bytes())
for name, patch in (("fixed", "fixed.patch"), ("specialized", "initial_specialization.patch")):
    with tempfile.TemporaryDirectory(dir=root / "temp", prefix="audit_fix_") as directory:
        target = Path(directory) / header
        target.parent.mkdir(parents=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"5c44fd70:{header}"], cwd=root))
        subprocess.run(["patch", "--silent", "-p1", "-i", str(dest / patch)], cwd=directory, check=True)
        restored = sha(target.read_bytes())
        if name == "fixed":
            assert restored == versions["fixed"]
        versions[name] = restored

timings = []
configs = guards = launches = 0
max_error = 0
for path in sorted(dest.glob("*.json")):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "configuration" not in data:
        continue
    label = path.name.split("_proxy_")[0].split("_assistant_")[0]
    version = ("old" if label.startswith("old") else "regressed" if label.startswith("regressed")
               else "fixed" if label.startswith("fixed") else "specialized")
    assert data["source_sha256"][header] == versions[version], path
    for name, expected in data["source_sha256"].items():
        if name != header:
            assert sha((root / name).read_bytes()) == expected, (path, name)
    cfg = data["configuration"]
    assert cfg["capacity"] == 4096 and cfg["passes"] == 7
    assert cfg["topk"] == 2048 and cfg["context"] == 65536
    assert len(data["results"]) == len(cfg["requests"])
    for row in data["results"]:
        configs += 1
        max_error = max(max_error, *row["correctness_max_abs_error"].values())
        assert all(error < .01 for error in row["correctness_max_abs_error"].values())
        for mode in cfg["modes"]:
            steps = row["per_step"][mode]
            assert len(steps) == row["steps"]
            assert all(len(step["samples_us"]) == cfg["passes"] for step in steps)
            launches += len(steps) * cfg["passes"]
            for index in range(cfg["passes"]):
                trials = [g for g in row["enqueue_guards"] if g["mode"] == mode and g["pass_index"] == index]
                accepted = sorted((g for g in trials if g["attempt"] == max(v["attempt"] for v in trials)),
                                  key=lambda g: g["first_step"])
                end = 0
                for guard in accepted:
                    assert guard["valid"] and guard["first_step"] == end
                    end = guard["last_step_exclusive"]
                    assert 0 < end - guard["first_step"] <= 32
                    guards += 1
                assert end == row["steps"]
    timings.append(dict(file=path.name, sha256=sha(path.read_bytes()), version=version,
                        configurations=len(data["results"])))

summary = json.loads((dest / "summary.json").read_text())
assert len(summary) == 110
large = [r for r in summary if r["query_rows"] > 32]
manifest = dict(
    base_commit="5c44fd70", source_sha256=versions,
    source_recovery={"old": "git show 109b1c76:" + header,
                     "regressed": "git show 5c44fd70:" + header,
                     "specialized": "5c44fd70 + initial_specialization.patch",
                     "fixed": "5c44fd70 + fixed.patch"},
    notes=["regressed_screen was originally mislabeled specialized after candidate generation aborted; "
           "the recorded source SHA identified the actual 5c44fd70 kernel. Renaming metadata is retained.",
           "specialized is the initial fix; fixed adds comments/formatting. All measurements are retained.",
           "f16/h64 abbreviates fetch_ctas=16/host_item_entries=64; performance inputs are BF16 with 64 heads.",
           "Kernel compiler dumps are archived as gzip; .cuh snapshots and redundant raw SASS are excluded.",
           "H800 production and pending third-round snapshot were not modified."],
    tests={"passed": 93, "seconds": 154.23, "log": "pytest_final.log"},
    coverage=dict(timing_files=len(timings), configurations=configs, accepted_enqueue_segments=guards,
                  timed_launches=launches, sampled_row_max_abs_error=max_error,
                  final_paired_configurations=55, final_paired_mode_cells=len(summary),
                  final_large_mode_cells=len(large),
                  worst_large_fixed_to_old=max(r["fixed_to_old"] for r in large)),
    timings=timings,
    compiler_artifacts={p.name: dict(sha256=sha(p.read_bytes()),
                                     raw_sha256=sha(gzip.decompress(p.read_bytes())))
                        for p in sorted(dest.glob("*_sass.txt.gz"))},
    resource_artifacts={p.name: sha(p.read_bytes()) for p in sorted(dest.glob("*_resources.txt"))})
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["coverage"], indent=2))
