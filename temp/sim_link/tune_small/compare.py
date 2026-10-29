"""Print same-parameter merge experiment comparisons; every slower case is retained."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("candidate")
parser.add_argument("baseline")
args = parser.parse_args()
dest = Path(__file__).resolve().parent


def steady(row, mode):
    steps = row["per_step"][mode][row["steps"] // 2:]
    return sum(s["median_us"] for s in steps) / len(steps)


for path in sorted(dest.glob(args.candidate + "_*.json")):
    candidate = json.loads(path.read_text())
    case = path.name.removeprefix(args.candidate + "_")
    old = json.loads((dest / (args.baseline + "_" + case)).read_text())
    assert candidate["configuration"] | {"output": None} == old["configuration"] | {"output": None}
    print(case)
    for new, before in zip(candidate["results"], old["results"]):
        for mode in ("ours_stage", "ours_inplace"):
            a, b = steady(before, mode), steady(new, mode)
            print(f"  q{new['requests']} {mode:12s} {a:8.2f} -> {b:8.2f} us ({(1-b/a)*100:+.1f}%)")
