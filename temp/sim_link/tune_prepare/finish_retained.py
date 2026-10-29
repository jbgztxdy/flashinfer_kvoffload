"""Archive final compiler evidence, generate all reports, then audit the complete experiment."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[3]
dest = Path(__file__).resolve().parent
while "retained verification complete" not in (dest / "retain.log").read_text():
    log = (dest / "retain.log").read_text()
    if "Traceback" in log:
        raise RuntimeError(log[-4000:])
    time.sleep(20)
for label, build in (("retained", "temp/jit_emu_prepare"), ("retained_normal", "temp/jit_prepare_normal")):
    library = next(p for p in (root / build).rglob("*_host_overlap_v2.so") if "dtype_q_bf16_" in p.name)
    (dest / f"{label}_resources.txt").write_bytes(subprocess.check_output(
        ["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-resource-usage", str(library)]))
    sass = subprocess.check_output(["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-sass", str(library)])
    with gzip.open(dest / f"{label}_sass.txt.gz", "wb") as stream:
        stream.write(sass)


def instructions(text):
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


old = instructions(gzip.decompress((dest / "base_sass.txt.gz").read_bytes()).decode())
new = instructions(gzip.decompress((dest / "retained_sass.txt.gz").read_bytes()).decode())
summary = [dict(stages=k[0], kv_tile=k[1], small=k[2] == "1", old_instructions=old[k][0],
                retained_instructions=v[0], instruction_text_equals_old=old[k][1] == v[1],
                instruction_sha256=v[1]) for k, v in new.items()]
(dest / "retained_compiler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for label in ("retained", "accepted", "final"):
    subprocess.run(["python3", str(dest / "report.py"), "--label", label], check=True)
subprocess.run(["python3", str(dest / "audit.py")], check=True)
print("retained reports and audit complete", flush=True)
