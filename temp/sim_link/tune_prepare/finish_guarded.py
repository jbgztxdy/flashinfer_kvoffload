"""Archive final enabled/default compiler outputs and compare them to measured baselines."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
while "guarded verification complete" not in (dest / "guarded_verify.log").read_text():
    log = (dest / "guarded_verify.log").read_text()
    if "Traceback" in log:
        raise RuntimeError(log[-4000:])
    time.sleep(20)


def functions(text):
    result = {}
    for part in re.split(r"\n\s*Function : ", text)[1:]:
        symbol = part.splitlines()[0].strip()
        if "BatchMLAHostBatchKernel" not in symbol:
            continue
        shape = re.search(r"KernelTraitsILb0ELj(\d+)ELb[01]ELj512ELj64ELj64ELj(\d+)", symbol)
        variant = re.search(r"ELb([01])EEEvT0_", symbol)
        ops = re.findall(r"/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*", part)
        result[(shape[1], shape[2], variant[1])] = (len(ops), hashlib.sha256("\n".join(ops).encode()).hexdigest())
    return result


old_emu = functions(gzip.decompress((dest / "base_sass.txt.gz").read_bytes()).decode())
old_normal = functions(gzip.decompress((root / "temp/sim_link/fix_large/fixed_normal_sass.txt.gz").read_bytes()).decode())
summary = {}
for label, build, baseline in (
        ("enabled", "temp/jit_emu_prepare_guarded", old_emu),
        ("enabled_normal", "temp/jit_prepare_guarded_normal", old_normal),
        ("default", "temp/jit_emu_prepare", old_emu),
        ("default_normal", "temp/jit_prepare_normal", old_normal)):
    library = next(p for p in (root / build).rglob("*_host_overlap_v2.so") if "dtype_q_bf16_" in p.name)
    resources = subprocess.check_output(["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-resource-usage", str(library)])
    (dest / f"{label}_resources.txt").write_bytes(resources)
    sass = subprocess.check_output(["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-sass", str(library)])
    with gzip.open(dest / f"{label}_sass.txt.gz", "wb") as stream:
        stream.write(sass)
    actual = functions(sass.decode())
    summary[label] = dict(library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                         kernels=[dict(stages=k[0], kv_tile=k[1], small=k[2] == "1", instructions=v[0],
                                       baseline_instructions=baseline[k][0],
                                       instruction_text_equals_baseline=baseline[k][1] == v[1],
                                       instruction_sha256=v[1]) for k, v in actual.items()])
(dest / "guarded_compiler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for label in ("enabled", "retained", "accepted", "final"):
    subprocess.run(["python3", str(dest / "report.py"), "--label", label], check=True)
subprocess.run(["python3", str(dest / "audit.py")], check=True)
print("guarded reports and audit complete", flush=True)
