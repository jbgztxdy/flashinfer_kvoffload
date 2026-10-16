# Purpose: Summarize batched MTP Memcpy / Direct Host attention results into Markdown and CSV.
# Usage: python temp/summarize_glm52_host_batch.py temp/host_attention_batch/random_mtp8.json
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = []
    for batch in data["results"]:
        baseline = next(c for c in batch["cases"] if c["host_fraction"] == 0)
        local_us = baseline["direct"]["median_us"]
        for case in batch["cases"]:
            direct = case["direct"]["median_us"]
            copy = case["copy"]["median_us"]
            rows.append(
                dict(
                    requests=batch["requests"],
                    query_rows=batch["query_rows"],
                    host_fraction=case["host_fraction"],
                    host_mib=case["host_ref_bytes"] / 2**20,
                    local_us=local_us,
                    copy_us=copy,
                    copy_p10_us=case["copy"]["p10_us"],
                    copy_p90_us=case["copy"]["p90_us"],
                    direct_us=direct,
                    direct_p10_us=case["direct"]["p10_us"],
                    direct_p90_us=case["direct"]["p90_us"],
                    copy_vs_local=copy / local_us,
                    direct_vs_local=direct / local_us,
                    direct_vs_copy=direct / copy,
                    direct_extra_gbps=(
                        case["host_ref_bytes"] / ((direct - local_us) * 1e3)
                        if direct > local_us and case["host_ref_bytes"]
                        else None
                    ),
                    cpu_pack_us=case["cpu_pack"]["median_us"],
                    max_abs_error=max(case["correctness_max_abs_error"].values()),
                )
            )
    csv_path = args.input.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fractions = sorted({row["host_fraction"] for row in rows})
    batches = sorted({(row["requests"], row["query_rows"]) for row in rows})
    lookup = {(row["requests"], row["host_fraction"]): row for row in rows}

    def table(title, value):
        lines = [f"### {title}", ""]
        lines.append(
            "| 请求数 | Query 行 | " + " | ".join(f"{f:.0%}" for f in fractions) + " |"
        )
        lines.append("|---:|---:|" + "---:|" * len(fractions))
        for requests, query_rows in batches:
            cells = [value(lookup[(requests, f)]) for f in fractions]
            lines.append(f"| {requests} | {query_rows} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def us(value):
        return f"{value:.1f}" if value < 1000 else f"{value / 1000:.2f} ms"

    text = [
        f"# 批量 MTP 下 Memcpy / Direct 结果（{data['device_name']}）",
        "",
        f"来源：`{args.input}`，commit `{data['git_commit'][:8]}`，{data['timestamp']}。",
        "",
        table("Memcpy GPU 耗时（H2D + attention）", lambda r: us(r["copy_us"])),
        table("Direct GPU 耗时", lambda r: us(r["direct_us"])),
        table("Memcpy 相对全本地", lambda r: f"{r['copy_vs_local']:.2f}×"),
        table("Direct 相对全本地", lambda r: f"{r['direct_vs_local']:.2f}×"),
        table("Direct / Memcpy", lambda r: f"{r['direct_vs_copy']:.2f}"),
        table("CPU gather（不计入 Memcpy GPU 时间）", lambda r: us(r["cpu_pack_us"])),
    ]
    md_path = args.input.with_suffix(".md")
    md_path.write_text("\n".join(text))
    print(f"wrote {csv_path} {md_path}")


if __name__ == "__main__":
    main()
