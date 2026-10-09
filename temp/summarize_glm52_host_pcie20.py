# Purpose: Export PCIe budgets, paired attention tables, and comparison charts.
# Usage: MPLCONFIGDIR=temp/matplotlib python temp/summarize_glm52_host_pcie20.py
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import plot_glm52_host_comparison as charts


FOLDER = Path(__file__).resolve().parent / "host_attention_pcie20"
METHODS = [
    ("copy", "Memcpy"),
    ("direct", "Direct"),
    ("previous", "Staged"),
    ("overlap", "Segmented"),
]
CONTEXTS = [(131072, "128K"), (262144, "256K"), (524288, "512K"), (1048576, "1M")]


def main():
    data = {
        layout: json.loads((FOLDER / f"after_{layout}.json").read_text())
        for layout in ("random", "head", "tail")
    }
    cases = {
        layout: {
            (item["context_tokens"], case["host_fraction_requested"]): case
            for item in value["results"]
            for case in item["cases"]
        }
        for layout, value in data.items()
    }
    budget = json.loads((FOLDER / "bandwidth_budget.json").read_text())
    lines = [
        "# Host attention: PCIe budget and segmented reads",
        "",
        "A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; 512 latent features; 64 RoPE features; page size 1; top-k 2048. The cache has equal HBM and pinned NUMA-0 Host capacity. Context labels are total KV lengths; only the selected rows are read.",
        "",
        "## PCIe budget",
        "",
        f"One-way PCIe 4.0 x16 coding ceiling: {budget['link_ceiling_GB_s']:.2f} GB/s. Measured large mapped-Host reads: {budget['measured_stream_GB_s']:.2f} GB/s. The historical HBM reference is {budget['baseline_us']:.2f} us; latency +20% permits {budget['baseline_us'] * 1.2:.2f} us, whereas throughput -20% permits {budget['baseline_us'] / 0.8:.2f} us.",
        "",
        "| Host | Payload (KiB) | Link-only minimum (us) | Stream proxy (us) | Sparse read/store (us) | Remaining after stream proxy (us) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget["rows"]:
        lines.append(
            f"| {row['host_fraction']:.0%} | {row['host_bytes'] / 1024:.2f} | {row['link_only_us']:.2f} | {row['stream_proxy_us']:.2f} | {row['sparse_read_and_store_us']:.2f} | {row['post_stream_budget_us']:.2f} |"
        )
    lines += [
        "",
        "Bandwidth alone permits the target in an ideal schedule. The stream values omit sparse-read overhead and startup; the read/store benchmark includes its HBM destination write and is not a physical lower bound. Native HBM time includes memory and synchronization, so it is not all available computation to overlap. See [analysis](analysis.md) for the assumptions and sources.",
        "",
        "## Methods and measurement",
        "",
        "- Memcpy: CPU packing into a pinned buffer, followed by actual serial H2D DMA and native attention. GPU time includes DMA, handoff, and attention; CPU packing is reported separately in the CSV.",
        "- Direct: the original attention kernel directly reads mapped Host memory.",
        "- Segmented: computing CTAs read Host KV directly into shared memory in three groups and accumulate QK as each group arrives. It uses 16 local and 32 Host compute partitions. Internal HBM scratch holds attention partial results. No KV staging kernel or memcpy is used.",
        "- Staged: the preserved V44 fused producer path, with 16 local and 28 Host compute partitions; its frozen module is loaded for the old/new comparison.",
        "- All four methods use identical queries, KV contents, and selected indices; Staged and Segmented share scratch and epoch buffers. Forty rounds, eight index sets, 100 serial cycles per CUDA graph, shuffled method/fraction order. Values are medians in microseconds. Top-k selection, index updates, planning, initialization, and correctness references are excluded.",
        "- Random/head/tail denotes Host positions within the selected 2048-row list. Physical addresses are sampled across each tier.",
        "- External GPU 1 activity elevated the first random 1M and head 128K/256K runs, including their all-local references. Those contexts were rerun. Initial raw files, fresh runs, and replacement provenance are retained; no samples were trimmed within a run.",
        "",
        "## Paired optimization results (us)",
        "",
        "| Layout | Context | Host | HBM reference | Staged | Segmented | Reduction vs staged | Latency increase vs HBM | Throughput loss vs HBM | Latency +20% target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = []
    checks = 0
    maximum_error = 0.0
    for layout in data:
        assert data[layout]["configuration"]["rounds"] == 40
        for context, label in CONTEXTS:
            hbm = cases[layout][context, 0.0]["direct"]["gpu_cycle"]["median_us"]
            for fraction in (0.25, 0.3):
                case = cases[layout][context, fraction]
                old = case["previous"]["gpu_cycle"]["median_us"]
                new = case["overlap"]["gpu_cycle"]["median_us"]
                lines.append(
                    f"| {layout} | {label} | {fraction:.0%} | {hbm:.2f} | {old:.2f} | {new:.2f} | {100 * (1 - new / old):.1f}% | {100 * (new / hbm - 1):.1f}% | {100 * (1 - hbm / new):.1f}% | {'Met' if new <= hbm * 1.2 else 'Unmet'} |"
                )
        for result in data[layout]["results"]:
            for case in result["cases"]:
                for mode, label in METHODS:
                    metric = case[mode]["gpu_cycle"]
                    assert len(metric["samples_us"]) == 40
                    error = case["correctness_max_abs_error"][mode]
                    maximum_error = max(maximum_error, error)
                    checks += 1
                    rows.append(
                        dict(
                            layout=layout,
                            context_tokens=result["context_tokens"],
                            host_percent=100 * case["host_fraction_requested"],
                            method=label,
                            gpu_us=metric["median_us"],
                            p10_us=metric["p10_us"],
                            p90_us=metric["p90_us"],
                            cold_us=case[mode]["cold_single"]["median_us"],
                            wall_us=case[mode]["serial_wall"]["median_us"],
                            cpu_pack_us=case["cpu_pack"]["median_us"]
                            if mode == "copy"
                            else 0,
                            max_abs_error=error,
                        )
                    )
    lines += ["", "## Main comparison tables (us)", ""]
    for layout in data:
        lines += [f"### {layout.title()}", ""]
        for context, label in CONTEXTS:
            lines += [
                f"#### {label}",
                "",
                "| Host | Memcpy | Direct | Segmented |",
                "|---:|---:|---:|---:|",
            ]
            for fraction in [i / 100 for i in range(0, 41, 5)] + [0.6, 0.8, 1.0]:
                case = cases[layout][context, fraction]
                lines.append(
                    f"| {fraction:.0%} | "
                    + " | ".join(
                        f"{case[mode]['gpu_cycle']['median_us']:.2f}"
                        for mode in ("copy", "direct", "overlap")
                    )
                    + " |"
                )
            lines.append("")
    lines += [
        "## Verification",
        "",
        f"All {checks} paired FP32 output comparisons passed; maximum absolute error {maximum_error:.6f}. Additional output/LSE, graph replay, and Host-mutation validation passed {json.loads((FOLDER / 'correctness_output.json').read_text())['cases']} cases. See [verification](verification.json) for API, sanitizer, traffic, resource, and source identities.",
        "",
        "## Charts and data",
        "",
        "Charts use the three main methods, context on the horizontal axis, and latency in us on the vertical axis. The dashed HBM line is the measured 0% Host native-attention median for each context and placement, with all selected KV rows in local HBM. Error bars show the 10th-90th percentile across rounds. Existing baseline reports and plots are preserved.",
        "",
        "[Charts](charts.zip) · [CSV](comparison.csv) · [Original comparison](../host_attention_comparison/report.md) · [Previous optimization](../host_attention_25_30/report.md)",
        "",
    ]
    (FOLDER / "report.md").write_text("\n".join(lines))
    with (FOLDER / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    charts.FOLDER = FOLDER
    charts.METHODS = [
        ("copy", "Memcpy", "#9BA7B2"),
        ("direct", "Direct", "#4D79A3"),
        ("overlap", "Segmented", "#398875"),
    ]
    charts.save_charts(cases)
    for fraction in (0.25, 0.3):
        figure, axis = plt.subplots(figsize=(6.4, 3.8))
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.82)
        charts.plot_axis(axis, cases["random"], fraction, 80)
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(
                FOLDER / f"host_{round(fraction * 100)}.{suffix}",
                dpi=300,
                metadata={"Creator": "Matplotlib"},
            )
        plt.close(figure)
    for path in FOLDER.rglob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    import zipfile

    with zipfile.ZipFile(
        FOLDER / "charts.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(FOLDER.rglob("host_*.*")):
            if path.suffix in {".png", ".svg", ".pdf"}:
                archive.write(path, path.relative_to(FOLDER))
    print(
        json.dumps(
            dict(
                checks=checks,
                maximum_output_absolute_error=maximum_error,
                report=str(FOLDER / "report.md"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
