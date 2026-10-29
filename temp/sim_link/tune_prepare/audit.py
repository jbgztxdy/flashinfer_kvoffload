"""Reconstruct experimental sources and validate data, guards, references, and unchanged emulator."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
header = "include/flashinfer/attention/mla_host_batch.cuh"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


base = subprocess.check_output(["git", "show", "f78ac492:" + header], cwd=root)
versions = {"base": sha(base)}
for patch in sorted(dest.glob("*.patch")):
    with tempfile.TemporaryDirectory(dir=root / "temp", prefix="prepare_audit_") as directory:
        target = Path(directory) / header
        target.parent.mkdir(parents=True)
        target.write_bytes(base)
        if patch.stem in ("final", "accepted", "retained", "guarded_full"):
            for name in ("tests/attention/test_mla_host_batch.py", "temp/h800/tiers.py"):
                target_file = Path(directory) / name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_bytes(subprocess.check_output(["git", "show", "ac507996:" + name], cwd=root))
        subprocess.run(["patch", "--silent", "-p1", "-i", str(patch)], cwd=directory, check=True)
        versions[patch.stem] = sha(target.read_bytes())
assert sha((root / header).read_bytes()) == versions["guarded"]
assert versions["guarded"] == versions["guarded_full"]
assert versions["accepted"] == versions["accepted_header"]
selected = []
in_option, active = False, True
for line in (root / header).read_text().splitlines(True):
    if line.strip() == "#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE":
        assert not in_option
        in_option, active = True, False
    elif in_option and line.strip() == "#else":
        active = True
    elif in_option and line.strip() == "#endif":
        in_option, active = False, True
    elif active:
        selected.append(line)
normalize = lambda s: re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", s))
assert not in_option and normalize("".join(selected)) == normalize(base.decode())
assert sha((root / "include/flashinfer/attention/far_link_emu.cuh").read_bytes()) == "5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095"
guard_count = launch_count = configs = 0
max_error = 0
timings = []
for path in sorted(dest.glob("*.json")):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "results" not in data or "configuration" not in data:
        continue
    label = path.name.split("_proxy_")[0].split("_assistant_")[0]
    version = ("base" if label.startswith(("base", "accepted_base")) else "guarded" if label.startswith(("enabled", "default"))
               else "retained" if label.startswith("retained")
               else "isolated" if label.startswith("isolated") else "accepted_header" if label.startswith(("accepted", "optin"))
               else "final" if label.startswith(("final", "floor")) else label)
    assert data["source_sha256"][header] == versions[version], path
    for name, expected in data["source_sha256"].items():
        if name != header:
            assert sha((root / name).read_bytes()) == expected, (path, name)
    cfg = data["configuration"]
    assert cfg["device"] == cfg["peer"] == 0
    assert cfg["passes"] == 7 and cfg["capacity"] == 4096
    assert cfg["topk"] == 2048 and cfg["context"] == 65536
    for row in data["results"]:
        configs += 1
        max_error = max(max_error, *row["correctness_max_abs_error"].values(), 0)
        assert all(error < .01 for error in row["correctness_max_abs_error"].values())
        for mode in cfg["modes"]:
            steps = row["per_step"][mode]
            assert len(steps) == row["steps"]
            assert all(len(s["samples_us"]) == cfg["passes"] for s in steps)
            launch_count += len(steps) * cfg["passes"]
            for index in range(cfg["passes"]):
                trials = [g for g in row["enqueue_guards"] if g["mode"] == mode and g["pass_index"] == index]
                accepted = sorted((g for g in trials if g["attempt"] == max(v["attempt"] for v in trials)),
                                  key=lambda g: g["first_step"])
                end = 0
                for guard in accepted:
                    assert guard["valid"] and guard["first_step"] == end
                    end = guard["last_step_exclusive"]
                    assert 0 < end - guard["first_step"] <= 32
                    guard_count += 1
                assert end == row["steps"]
    timings.append(dict(file=path.name, sha256=sha(path.read_bytes()), version=version,
                        configurations=len(data["results"])))
paired = json.loads((dest / "summary.json").read_text())
assert len(paired) == 88
coverage = dict(timing_files=len(timings), configurations=configs, accepted_enqueue_segments=guard_count,
                timed_launches=launch_count, sampled_row_max_abs_error=max_error,
                final_paired_configurations=44, final_paired_mode_cells=len(paired))
manifest = dict(base_commit="ac507996", source_sha256=versions, coverage=coverage, timings=timings,
                default_source_tokens_equal_baseline=True,
                captures={str(p.relative_to(root)): file_sha(p) for pattern in
                          ("temp/mtp_overlap/rounds_ctx65536_*.pt", "temp/glm_trace/assistant64k/doc*.pt")
                          for p in sorted(root.glob(pattern))},
                tests=dict(passed=101, seconds=175.76, log="pytest_guarded.log", retained_log="pytest_retained.log", pre_isolation_log="pytest_accepted.log", pipeline_experiment_log="pytest_final.log"),
                compiler_artifacts={p.name: dict(sha256=sha(p.read_bytes()),
                                                  raw_sha256=sha(gzip.decompress(p.read_bytes())))
                                    for p in sorted(dest.glob("*_sass.txt.gz"))},
                resource_artifacts={p.name: sha(p.read_bytes()) for p in sorted(dest.glob("*_resources.txt"))},
                notes=["GPU0/1 occupied by unrelated services. Only free GPU3 A100-PCIE used.",
                       "Simulated far and near are separate same-GPU HBM allocations, not physical NVLink. "
                       "Unchanged emulator charges the same bytes and latency. Physical Host cases use normal builds.",
                       "The two-version controlled baseline uses the same tiers.py same-device access-list fix.",
                       "Profile patches and timestamp traces are diagnostic, never final timing evidence.",
                       "16-row fetch ring regressed. The 32-row compact fetch ring regressed 15% on one "
                       "zero-latency case. It is only in saved experimental patches, absent from the retained kernel.",
                       "final_* files measure the pipeline candidate; accepted_* and retained_* measure intermediate candidates. "
                       "An apparent recovery after isolating helpers did not persist; accepted/retained kernel instruction "
                       "text was identical. No claim that isolation fixed the Host regression is supported.",
                       "enabled_* and default_* compare the final explicit SMALL_PREPARE option with the original default. "
                       "The option is disabled by default after a 6% Host timing regression in intermediate candidates.",
                       "LRU ties within a launch may pick different victims, so reports preserve miss counts.",
                       "All lower bounds use normal-build same-GPU pure-HBM samples; emulator floors are ignored."])
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(coverage, indent=2))
