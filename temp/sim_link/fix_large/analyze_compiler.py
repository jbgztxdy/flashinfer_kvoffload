"""Summarize and archive the measured SASS/resource dumps without claiming byte identity."""
import gzip
import hashlib
import json
from pathlib import Path
import re

dest = Path(__file__).resolve().parent
summary = {}
for label in ("old", "regressed", "fixed", "old_normal", "fixed_normal"):
    sass_path = dest / f"{label}_sass.txt"
    resources = (dest / f"{label}_resources.txt").read_text()
    text = sass_path.read_text()
    rows = []
    for part in re.split(r"\n\s*Function : ", text)[1:]:
        symbol = part.splitlines()[0].strip()
        if "BatchMLAHostBatchKernel" not in symbol:
            continue
        instructions = re.findall(r"/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*", part)
        resource = re.search(r"Function " + re.escape(symbol) + r":?\s+([^\n]+)", resources)
        assert resource
        variant = re.search(r"ELb([01])EEEvT0_", symbol)
        shape = re.search(r"KernelTraitsILb0ELj(\d+)ELb[01]ELj512ELj64ELj64ELj(\d+)", symbol)
        rows.append(dict(symbol=symbol, stages=int(shape[1]), kv_tile=int(shape[2]),
                         row_merge=bool(int(variant[1])) if variant else None,
                         instructions=len(instructions), resource=resource[1],
                         instruction_sha256=hashlib.sha256("\n".join(instructions).encode()).hexdigest()))
    with gzip.open(sass_path.with_suffix(".txt.gz"), "wb") as stream:
        stream.write(text.encode())
    summary[label] = rows
for kind in ("", "_normal"):
    before = {(r["stages"], r["kv_tile"]): r for r in summary["old" + kind]}
    for row in summary["fixed" + kind]:
        if not row["row_merge"]:
            old = before[(row["stages"], row["kv_tile"])]
            row["instruction_count_equals_old"] = row["instructions"] == old["instructions"]
            row["instruction_text_equals_old"] = row["instruction_sha256"] == old["instruction_sha256"]
(dest / "compiler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("compiler evidence archived", flush=True)
