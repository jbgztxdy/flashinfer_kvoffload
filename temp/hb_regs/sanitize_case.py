# Purpose: Small batched Host MLA launch for compute-sanitizer (split tail rows and full rows).
# Usage: compute-sanitizer --tool racecheck python temp/hb_regs/sanitize_case.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_glm52_decode  # noqa: E402,F401  (points JIT at the source tree)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "attention"))
from test_mla_host_batch import _run  # noqa: E402

for requests, mtp in ((2, 4), (16, 8)):
    _run(requests, mtp, 64, 512, 0.3, shared=0.5, repeats=2)
print("ok")
