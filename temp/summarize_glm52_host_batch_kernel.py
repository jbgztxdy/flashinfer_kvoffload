# Purpose: Tables for the Memcpy / Direct / batched Host kernel comparison.
# Usage: python temp/summarize_glm52_host_batch_kernel.py temp/host_attention_batch_kernel/mtp.json
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.inputs:
        data = json.loads(path.read_text())
        rows = []
        for batch in data["results"]:
            for case in batch["cases"]:
                row = dict(
                    requests=batch["requests"],
                    query_rows=batch["query_rows"],
                    host_fraction=case["host_fraction"],
                    host_fraction_actual=case["host_fraction_actual"],
                    unique_ratio=case["unique_ratio"],
                    host_unique_mib=case["host_unique_bytes_median"] / 2**20,
                    cpu_pack_us=case["cpu_pack"]["median_us"],
                    max_abs_error=max(case["correctness_max_abs_error"].values(), default=None),
                )
                for mode in ("direct", "copy", "batch"):
                    row[f"{mode}_us"] = case[mode]["median_us"]
                    row[f"{mode}_p10_us"] = case[mode]["p10_us"]
                    row[f"{mode}_p90_us"] = case[mode]["p90_us"]
                row["batch_vs_copy"] = row["batch_us"] / row["copy_us"]
                row["batch_vs_direct"] = row["batch_us"] / row["direct_us"]
                rows.append(row)
        with path.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fractions = sorted({r["host_fraction"] for r in rows})
        batches = sorted({(r["requests"], r["query_rows"]) for r in rows})
        lookup = {(r["requests"], r["host_fraction"]): r for r in rows}

        def us(v):
            return f"{v:.1f}" if v < 1000 else f"{v / 1000:.2f} ms"

        def table(title, cell):
            lines = [f"### {title}", "", "| 请求数 | Query 行 | " + " | ".join(f"{f:.0%}" for f in fractions) + " |",
                     "|---:|---:|" + "---:|" * len(fractions)]
            for requests, query_rows in batches:
                lines.append(f"| {requests} | {query_rows} | "
                             + " | ".join(cell(lookup[(requests, f)]) for f in fractions) + " |")
            return "\n".join(lines) + "\n"

        def best(r):
            mark = "" if r["batch_us"] < min(r["copy_us"], r["direct_us"]) else " ✗"
            return f"{us(r['batch_us'])}{mark}"

        losses = [r for r in rows if r["batch_us"] >= min(r["copy_us"], r["direct_us"])]
        text = [
            f"# Memcpy / Direct / Batched Host（{data['device_name']}, top-k {data['configuration']['index_mode']}）",
            "",
            f"来源：`{path}`，commit `{data['git_commit'][:8]}`{'（含未提交改动）' if data.get('git_dirty') else ''}，"
            f"{data['timestamp']}，fetch CTA {data['configuration']['fetch_ctas']}。",
            "",
            f"Batched 不快于两者中较快者的格子：{len(losses)} / {len(rows)}（表中标 ✗）。",
            "",
            table("Batched GPU 耗时", best),
            table("Memcpy GPU 耗时（H2D + attention，CPU gather 不计）", lambda r: us(r["copy_us"])),
            table("Direct GPU 耗时", lambda r: us(r["direct_us"])),
            table("Batched / Memcpy", lambda r: f"{r['batch_vs_copy']:.2f}"),
            table("Batched / Direct", lambda r: f"{r['batch_vs_direct']:.2f}"),
            table("实际 Host 命中率", lambda r: f"{r['host_fraction_actual']:.3f}"),
            table("去重后 Host 页 / Host 引用", lambda r: f"{r['unique_ratio']:.2f}" if r["host_fraction"] else "-"),
            table("CPU gather（Memcpy 需要，不计入其 GPU 时间）", lambda r: us(r["cpu_pack_us"])),
        ]
        path.with_suffix(".md").write_text("\n".join(text))
        print(f"wrote {path.with_suffix('.md')} ({len(losses)} non-wins)")


if __name__ == "__main__":
    main()
