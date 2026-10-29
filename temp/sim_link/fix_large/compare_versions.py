"""Compare both previous headers to the fixed one, saving exact compiler resources."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
os.chdir(root)
dest = Path(__file__).resolve().parent
header = Path("include/flashinfer/attention/mla_host_batch.cuh")
fixed = header.read_bytes()
(dest / "fixed.cuh").write_bytes(fixed)


def resources(label):
    library = next(path for path in Path("temp/jit_emu").rglob("*.so")
                   if path.name.endswith("_host_overlap_v2.so"))
    for kind, flag in (("resources", "--dump-resource-usage"), ("sass", "--dump-sass")):
        with (dest / f"{label}_{kind}.txt").open("w") as output:
            subprocess.run(["/usr/local/cuda-13.1/bin/cuobjdump", flag, str(library)],
                           stdout=output, check=True)


try:
    for label, ref in (("old", "109b1c76"), ("regressed", "5c44fd70")):
        header.write_bytes(subprocess.check_output(["git", "show", ref + ":" + str(header)]))
        subprocess.run(["python3", str(dest / "screen.py"), label], check=True)
        resources(label)
    header.write_bytes(fixed)
    subprocess.run(["python3", str(dest / "screen.py"), "fixed", "--full"], check=True)
    resources("fixed")
finally:
    header.write_bytes(fixed)
print("all versions complete; fixed header restored", flush=True)
