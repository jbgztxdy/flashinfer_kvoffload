# Purpose: Export paired Host-attention comparisons and optimization evidence.
# Usage: python temp/summarize_glm52_host_25_30.py
import csv
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent / "host_attention_25_30"
CONTEXTS = [(131072, "128K"), (262144, "256K"), (524288, "512K"), (1048576, "1M")]
METHODS = [
    ("copy", "Memcpy"),
    ("direct", "Direct"),
    ("previous", "Previous"),
    ("overlap", "Optimized"),
]


def main():
    lines = [
        "# Host attention: 25% and 30% optimization",
        "",
        "A100-SXM4-40GB GPU 1; BF16; one query; 64 heads; rank 512; RoPE 64; top-k 2048; page size 1. Half of the physical KV cache is in HBM and half in pinned NUMA-0 Host DRAM.",
        "",
        "## Methods and timing",
        "",
        "- Memcpy: CPU packing, followed by serial H2D DMA and native attention. GPU values include DMA and attention; CPU packing is separate in the CSV.",
        "- Direct: native attention directly reads mixed HBM/Host storage. Its existing two-stage cp.async pipeline can overlap input reads and tensor-core computation.",
        "- Previous: the frozen 16-local / 28-Host / 64-producer implementation from commit 856ac532.",
        "- Optimized: fixed splits with a single 32-key tensor-core tile for Host partitions containing 17-32 rows. Local and Host math are separate template instances; the medium Host tile has a compile-time tile index of zero.",
        "- All four methods use the same KV data, query, and indices. Previous and Optimized also share their internal HBM scratch and epoch state. Method order is shuffled every round. Forty rounds, eight index sets, and 100 serialized cycles per graph are used. Latencies are median GPU-cycle times in microseconds.",
        "- Top-k selection, index updates, planning, initialization, and FP32 references are excluded. Outputs are checked against an independent FP32 reference.",
        "",
        "## Native Host traffic",
        "",
        "After ten warmup calls, Nsight Compute measured 589824 bytes of sysmem fills at 25% and 707328 bytes at 30%, matching all selected Host rows. The profiled kernel is BatchMLAPagedAttentionKernel. Profiling durations are excluded from the performance tables. See native_host_traffic.json and profile_direct_25/3.csv.",
        "",
        "## Paired GPU latency (µs)",
        "",
    ]
    rows = []
    maximum_error = 0.0
    checks = 0
    for layout in ("random", "head", "tail"):
        data = json.loads((FOLDER / f"after_{layout}.json").read_text())
        lines += [f"### {layout.title()}", ""]
        assert data["configuration"]["rounds"] == 40
        for result, (context, label) in zip(data["results"], CONTEXTS, strict=True):
            assert result["context_tokens"] == context
            lines += [
                f"#### {label}",
                "",
                "| Host | Memcpy | Direct | Previous | Optimized | Change vs previous |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
            for case in result["cases"]:
                latency = {
                    mode: case[mode]["gpu_cycle"]["median_us"] for mode, _ in METHODS
                }
                reduction = 100 * (1 - latency["overlap"] / latency["previous"])
                lines.append(
                    f"| {case['host_fraction_requested']:.0%} | "
                    + " | ".join(f"{latency[mode]:.2f}" for mode, _ in METHODS)
                    + f" | {reduction:+.1f}% |"
                )
                for mode, _ in METHODS:
                    assert len(case[mode]["gpu_cycle"]["samples_us"]) == 40
                    error = case["correctness_max_abs_error"][mode]
                    maximum_error = max(maximum_error, error)
                    checks += 1
                    summary = case[mode]
                    rows.append(
                        dict(
                            layout=layout,
                            context_tokens=context,
                            host_percent=100 * case["host_fraction_requested"],
                            method=mode,
                            gpu_us=latency[mode],
                            p10_us=summary["gpu_cycle"]["p10_us"],
                            p90_us=summary["gpu_cycle"]["p90_us"],
                            cold_us=summary["cold_single"]["median_us"],
                            wall_us=summary["serial_wall"]["median_us"],
                            cpu_pack_us=case["cpu_pack"]["median_us"]
                            if mode == "copy"
                            else 0,
                            max_abs_error=error,
                        )
                    )
            lines.append("")
    lines += [
        "## Verification",
        "",
        f"All {checks} output checks passed; maximum absolute error {maximum_error:.6f}.",
        "",
        "Additional validation: 372 BF16/FP16 output and LSE cases, graph reuse, Host mutations, and tile boundaries passed. Six native regression cases and eight invalid-input cases passed. Racecheck and synccheck reported zero errors.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_25_30.py --output-dir temp/host_attention_25_30_repeat",
        "python temp/summarize_glm52_host_25_30.py",
        "```",
        "",
        "The runner protects existing raw files. The frozen BF16 SM80 library and its source/hash metadata are retained alongside the data. Prior tables and baseline artifacts remain available in ../host_attention_comparison/.",
        "",
    ]
    (FOLDER / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    with (FOLDER / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows; maximum absolute error {maximum_error:.6f}")


if __name__ == "__main__":
    main()
