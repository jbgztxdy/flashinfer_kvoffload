# Purpose: Summarize complete native and mixed-tier MLA overlap benchmarks.
# Usage: python temp/summarize_glm52_host_overlap.py
import json
from pathlib import Path
import statistics

folder = Path(__file__).resolve().parent
layouts = ("random", "head", "tail")
measurements = {
    mode: {
        layout: json.loads(
            (folder / f"glm52_host_{mode}_final_{layout}.json").read_text()
        )
        for layout in layouts
    }
    for mode in ("native", "overlap")
}
fractions = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
rows = []
for fraction in fractions:
    row = {"host_fraction": fraction, "host_tokens": round(fraction * 2048)}
    row["host_bytes"] = row["host_tokens"] * 1152
    row["transfer_us_at_streaming_27_GB_s"] = row["host_bytes"] / 27000
    row["transfer_us_at_pcie_31_5077_GB_s"] = row["host_bytes"] / 31507.6923
    for mode, by_layout in measurements.items():
        samples = {timing: [] for timing in ("hot_graph", "cold_single_graph", "eager")}
        overheads = []
        for data in by_layout.values():
            assert len(data["results"]) == 4
            for context in data["results"]:
                case = next(
                    c
                    for c in context["cases"]
                    if c["host_fraction_requested"] == fraction
                )
                for timing in samples:
                    samples[timing].append(case[timing]["median_us"])
                overheads.append((case["hot_graph"]["relative_to_hbm"] - 1) * 100)
        row[mode] = {
            timing: {
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
            }
            for timing, values in samples.items()
        }
        row[mode]["hot_overhead_percent_median"] = statistics.median(overheads)
        row[mode]["hot_overhead_percent_max"] = max(overheads)
    rows.append(row)
summary = {
    "rows": rows,
    "acceptance_overhead_percent": 5,
    "fully_met_at_20_percent": all(
        r["overlap"]["hot_overhead_percent_max"] <= 5
        for r in rows
        if r["host_fraction"] <= 0.2
    ),
}
(folder / "glm52_host_overlap_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
lines = [
    "# Mixed HBM and pinned Host DRAM MLA overlap experiment",
    "",
    "## Scope and measured outcome",
    "",
    "A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; latent rank 512; RoPE 64; page size one; top-k 2048. Total cache contexts: 131072, 262144, 524288, 1048576 tokens. The first physical half resides in GPU HBM and the second in pinned NUMA-0 Host DRAM, mapped into one CUDA VMM address range. Host selections occupy random, head, or tail positions among the selected 2048 keys; physical token addresses remain sampled across each tier.",
    "",
    "The acceptance assumption is at most 5% latency increase relative to the same operator with 0% Host hits. This is a benchmark result, not a claim that every Host transfer is hidden. The table aggregates the median of 12 independently measured context/layout medians; the range and per-case results below retain variation.",
    "",
    "## Latency and transfer model",
    "",
    "| Host hits | Host keys | Native direct (us) | Fused overlap (us) | Fused / its 0% | Transfer at 27 GB/s (us) |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['host_fraction']:.0%} | {row['host_tokens']} | {row['native']['hot_graph']['median_us']:.2f} | {row['overlap']['hot_graph']['median_us']:.2f} | +{row['overlap']['hot_overhead_percent_median']:.1f}% | {row['transfer_us_at_streaming_27_GB_s']:.2f} |"
    )
lines += [
    "",
    "The 27 GB/s number is measured mapped-memory streaming bandwidth. The PCIe 4.0 x16 coding-adjusted one-way ceiling is 31.51 GB/s. The transfer columns omit launch, sparse addressing, synchronization, and attention arithmetic. The isolated sparse-fetch microbenchmark takes approximately 21 us at 20% (410 rows, 472320 bytes), so the 17.49 us streaming estimate is optimistic for this workload. Neither figure proves complete overlap.",
    "",
    "## Implementation and remaining tail",
    "",
    "The opt-in `host_page_start` path adds one fused cooperative CUDA attention kernel. It partitions selected keys into 16 local compute tasks and 28 Host compute tasks. The other 64 CTAs use one warp group to issue independent packed PCIe reads and another to merge local attention states. Epoch-tagged ready flags publish staged rows and partial attention results with release/acquire ordering. Host attention uses tensor cores and a one-tile path for small chunks. CPU values retain their original BF16/FP16 representation.",
    "",
    "The kernel reads mapped Host memory directly, then writes those rows into internal HBM scratch before tensor-core attention consumes them. This buffering happens inside attention. There is no external gather kernel or CUDA memcpy in the timed path. Internal allocation is 5.25 MiB of HBM plus 32 KiB synchronization state, allocated and initialized during warmup. It is not a GPU tensor allocation that happens to contain previously copied CPU data.",
    "",
    "Phase instrumentation in `glm52_overlap_host_v22_trace.json` shows that local attention and CPU reads overlap. At 20%, final Host row readiness occurs around 21 us, followed by Host QK/PV, output writes, and partial-state reduction. Instrumentation is a separate debug run and is not used as the final latency measurement. A write-combined Host allocation was also measured; it did not materially improve sparse transfer time (`sparse_host_allocators.json`).",
    "",
    "The trial history keeps rejected designs, including scalar final PV, duplicated QK, smaller/larger tiles, staged probability sharing, and parallel partial reductions. The final implementation restores the best balanced packed-load design and adds explicit warp synchronization around shared output conversion. Racecheck originally identified warp-local shared-memory write/read hazards in the reused native output helper; the Host path now synchronizes before reading and before reusing that tile. Native calls retain their original default path.",
    "",
    "## Timing and correctness",
    "",
    "Each case uses 40 rounds, eight distinct index sets, 100 attention calls per hot CUDA Graph, 20 eager calls, and a separate one-call measurement after 64 MiB L2 eviction. Host fractions are shuffled in each round. Top-k selection, plan, index updates, cache initialization, graph construction, and independent FP32 references are excluded. Host reads, HBM staging, attention computation, and output merging are included. Eager GPU events also contain CPU launch gaps; cold single-call timing includes graph launch gaps.",
    "",
    "`glm52_overlap_correctness.json` records 228 independent FP32 comparisons covering BF16/FP16, all fractions, random/head/tail layouts, two seeds, base-2/base-e LSE, graph reuse, and CPU-side Host mutations. `glm52_overlap_api_correctness.json` records native page-size-16, causal, multi-request, multi-query, 128-head regressions and rejected invalid Host inputs. Sanitizer logs are retained separately. The nine historical baseline hashes are preserved in `glm52_host_historical_baselines_manifest.json`.",
    "",
    "GPU 1 was selected for serial measurements. Other GPUs ran unrelated jobs, and GPU 1 contained two idle EngineCore processes. Before/after GPU snapshots, source hashes, and the Git commit are stored in each result. Hardware was shared; these results do not establish single-tenant production behavior or H800 performance.",
    "",
    "## Per-context and layout results",
    "",
]
for layout in layouts:
    data = measurements["overlap"][layout]
    lines += [
        f"### {layout}",
        "",
        "| Context | Host | Hot (us) | Increase | Cold single (us) | Eager (us) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for context in data["results"]:
        for case in context["cases"]:
            lines.append(
                f"| {context['context_tokens']} | {case['host_fraction_actual']:.1%} | {case['hot_graph']['median_us']:.2f} | {(case['hot_graph']['relative_to_hbm'] - 1) * 100:.1f}% | {case['cold_single_graph']['median_us']:.2f} | {case['eager']['median_us']:.2f} |"
            )
    lines.append("")
lines += [
    "## Reproduce",
    "",
    "```bash",
    "PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_overlap_benchmarks.py",
    "python temp/summarize_glm52_host_overlap.py",
    "```",
    "",
    "Existing benchmark outputs are protected from overwrite. Use new output paths for a rerun.",
]
(folder / "glm52_host_overlap_report.md").write_text("\n".join(lines) + "\n")
print(
    json.dumps(
        {
            "fully_met_at_20_percent": summary["fully_met_at_20_percent"],
            "table": [
                {
                    "host": r["host_fraction"],
                    "native_us": r["native"]["hot_graph"]["median_us"],
                    "overlap_us": r["overlap"]["hot_graph"]["median_us"],
                    "increase_percent": r["overlap"]["hot_overhead_percent_median"],
                }
                for r in rows
            ],
        },
        indent=2,
    )
)
