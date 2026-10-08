# Purpose: Export Host-attention tables and grouped latency charts.
# Usage: MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_comparison.py
import argparse
import csv
import json
import math
from pathlib import Path
import re
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "temp/host_attention_comparison"
CONTEXTS = [131072, 262144, 524288, 1048576]
TICKS = ["128K", "256K", "512K", "1M"]
FRACTIONS = [i / 100 for i in range(0, 41, 5)]
METHODS = [
    ("copy", "Memcpy", "#9BA7B2"),
    ("direct", "Direct", "#4D79A3"),
    ("overlap", "Overlap", "#398875"),
]
LAYOUTS = ["random", "head", "tail"]


def load(layout):
    data = json.loads((FOLDER / f"raw_{layout}.json").read_text())
    return data, {
        (result["context_tokens"], case["host_fraction_requested"]): case
        for result in data["results"]
        for case in result["cases"]
    }


def table(cases, context, metric="gpu_cycle", fractions=FRACTIONS):
    lines = [
        "| Host | " + " | ".join(label for _, label, _ in METHODS) + " |",
        "|---:|---:|---:|---:|",
    ]
    for fraction in fractions:
        values = [
            cases[(context, fraction)][method][metric]["median_us"]
            for method, _, _ in METHODS
        ]
        lines.append(
            f"| {fraction:.0%} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )
    return lines


def packing_table(cases):
    lines = ["| Host | " + " | ".join(TICKS) + " |", "|---:|---:|---:|---:|---:|"]
    for fraction in FRACTIONS:
        values = [
            cases[(context, fraction)]["cpu_pack"]["median_us"] for context in CONTEXTS
        ]
        lines.append(
            f"| {fraction:.0%} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + " |"
        )
    return lines


def plot_axis(axis, cases, fraction, ymax, legend=True, ylabel=True):
    x = np.arange(len(CONTEXTS))
    width = 0.235
    for index, (method, label, color) in enumerate(METHODS):
        summaries = [
            cases[(context, fraction)][method]["gpu_cycle"] for context in CONTEXTS
        ]
        median = np.array([row["median_us"] for row in summaries])
        errors = np.array(
            [
                [row["median_us"] - row["p10_us"] for row in summaries],
                [row["p90_us"] - row["median_us"] for row in summaries],
            ]
        )
        axis.bar(
            x + (index - 1) * width,
            median,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            yerr=errors,
            error_kw=dict(ecolor="#4B535A", elinewidth=0.75, capsize=2, capthick=0.75),
        )
    axis.set_xticks(x, TICKS)
    axis.set_xlabel("Context", labelpad=8)
    if ylabel:
        axis.set_ylabel("Latency (µs)", labelpad=8)
    axis.set_ylim(0, ymax)
    axis.set_yticks(np.arange(0, ymax + 1, 10))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E6E9EC", linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#ADB4BA")
        axis.spines[spine].set_linewidth(0.7)
    axis.tick_params(length=0, pad=6)
    axis.set_title(
        f"{fraction:.0%} Host", loc="left", pad=18, fontsize=12, fontweight="medium"
    )
    if legend:
        axis.legend(
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1.01, 1.165),
            frameon=False,
            handlelength=1.15,
            handletextpad=0.5,
            columnspacing=1.4,
            borderaxespad=0,
        )


def save_charts(cases_by_layout):
    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    ymax = (
        math.ceil(
            max(
                cases[(context, fraction)][method]["gpu_cycle"]["p90_us"]
                for cases in cases_by_layout.values()
                for context in CONTEXTS
                for fraction in (0.1, 0.15, 0.2)
                for method, _, _ in METHODS
            )
            / 10
        )
        * 10
    )
    for layout, cases in cases_by_layout.items():
        directory = FOLDER if layout == "random" else FOLDER / layout
        directory.mkdir(exist_ok=True)
        for fraction in (0.1, 0.15, 0.2):
            figure, axis = plt.subplots(figsize=(6.4, 3.8))
            figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.82)
            plot_axis(axis, cases, fraction, ymax)
            for suffix in ("png", "svg", "pdf"):
                figure.savefig(
                    directory / f"host_{round(fraction * 100)}.{suffix}",
                    dpi=300,
                    metadata={"Creator": "Matplotlib"},
                )
            plt.close(figure)
    cases = cases_by_layout["random"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.55), sharey=True)
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.19, top=0.79, wspace=0.16)
    for index, (axis, fraction) in enumerate(zip(axes, (0.1, 0.15, 0.2), strict=True)):
        plot_axis(axis, cases, fraction, ymax, legend=False, ylabel=index == 0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        frameon=False,
        handlelength=1.2,
        columnspacing=2.0,
    )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            FOLDER / f"host_10_15_20.{suffix}",
            dpi=300,
            metadata={"Creator": "Matplotlib"},
        )
    plt.close(figure)
    for path in FOLDER.rglob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    with zipfile.ZipFile(
        FOLDER / "charts.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(FOLDER.rglob("host_*.*")):
            if path.suffix in {".png", ".svg", ".pdf"}:
                archive.write(path, path.relative_to(FOLDER))


def history():
    lines = [
        "# Historical Host-attention measurements",
        "",
        "These exploratory results are retained for traceability. They used earlier implementations and measurement configurations. The comparison figures use the matched measurements in raw_random/head/tail.json.",
        "",
        "## Operator variants",
        "",
        "| Variant | 0% | 10% | 15% | 20% | Artifact |",
        "|---|---:|---:|---:|---:|---|",
    ]
    paths = list((ROOT / "temp").glob("glm52_overlap_host_v*_random.json"))
    paths.sort(key=lambda path: int(re.search(r"_v(\d+)", path.name)[1]))
    paths = [ROOT / "temp/glm52_overlap_stage4_random.json"] + paths
    for path in paths:
        data = json.loads(path.read_text())
        result = next(
            row for row in data["results"] if row["context_tokens"] == 1048576
        )
        cases = {row["host_fraction_requested"]: row for row in result["cases"]}
        label = re.search(r"_v(\d+)", path.name)
        label = f"V{label[1]}" if label else "Stage4"
        values = [cases[f]["hot_graph"]["median_us"] for f in (0, 0.1, 0.15, 0.2)]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{v:.2f}" for v in values)
            + f" | [{path.stem}](../{path.name}) |"
        )
    lines += [
        "",
        "## Legacy staging script",
        "",
        "The historical script named host_memcpy exposes pinned CPU memory as a CUDA tensor. Its indexed read is a GPU gather from mapped Host memory followed by a GPU copy. It is a different staging baseline from the actual H2D DMA measured here. Its metadata and timing labels also differ, and it ran on GPU 0 with fewer rounds. The original files are preserved unchanged; the legacy values below are not used in the comparison charts.",
        "",
    ]
    data = json.loads(
        (ROOT / "temp/glm52_sparse_decode_host_memcpy_full.json").read_text()
    )
    for result in data["results"]:
        lines += [
            f"### Context {result['context_tokens']}",
            "",
            "| Host | Gather/copy (µs) | Single attention (µs) | Repeated attention (µs) |",
            "|---:|---:|---:|---:|",
        ]
        for row in result["cases"]:
            lines.append(
                f"| {row['remote_fraction_requested']:.0%} | {row['copy']['median_us']:.2f} | {row['cold']['median_us']:.2f} | {row['hot']['median_us']:.2f} |"
            )
        lines.append("")
    (FOLDER / "history.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--plots-only", action="store_true")
    outputs.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    loaded = {
        layout: load(layout)
        for layout in LAYOUTS
        if (FOLDER / f"raw_{layout}.json").exists()
    }
    cases_by_layout = {layout: cases for layout, (_, cases) in loaded.items()}
    if args.plots_only:
        save_charts(cases_by_layout)
        return
    if list(loaded) != LAYOUTS:
        raise ValueError("Complete random, head, and tail measurements are required")
    for _data, cases in loaded.values():
        assert len(cases) == len(CONTEXTS) * 12
        for case in cases.values():
            assert set(case["correctness_max_abs_error"]) == {
                "copy",
                "direct",
                "overlap",
            }
            for mode, _, _ in METHODS:
                assert len(case[mode]["gpu_cycle"]["samples_us"]) == 40
    if not args.tables_only:
        save_charts(cases_by_layout)
    lines = [
        "# Host attention: memcpy, direct access, and overlap",
        "",
        "A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; compressed rank 512; RoPE 64; top-k 2048; page size 1. A single attention layer is measured. Context labels are powers of two: 128K=131072, 256K=262144, 512K=524288, 1M=1048576.",
        "",
        "Half of the physical cache resides in local HBM and half in pinned NUMA-0 Host memory. Hit fractions apply to the selected 2048 rows and are rounded to the nearest row. Random/head/tail describes their positions in the selected list; physical addresses are sampled across each tier.",
        "",
        "## Measurement",
        "",
        "- Memcpy: gather selected Host rows on CPU into a pinned buffer, then H2D DMA and native attention on one stream. Every repeated GPU cycle performs its own DMA. Local HBM is aliased directly, with no local gather or copy.",
        "- Direct: native attention directly reads mapped Host memory.",
        "- Overlap: the optimized fused attention kernel fetches Host rows concurrently with local computation.",
        "- Main tables and charts report the median GPU cycle in µs. Memcpy includes DMA, its handoff/dependency cost, and attention. CPU packing is excluded from these GPU values and reported separately in details.md; serial wall time includes packing and synchronization.",
        "- Each case uses the same KV data, query, and eight index sets across all three methods. Forty rounds are measured, with 100 serialized cycles per CUDA graph. Fraction and method order are shuffled. Error bars show the 10th-90th percentiles across rounds.",
        "- Top-k selection, index updates/remapping, plan, initialization, and FP32 references are excluded. All measured cases passed independent FP32 comparisons.",
        "- The 20% memcpy trace contains 100 actual pinned-Host-to-device copies and 100 native attention kernels on the same stream, with no overlap and no GPU gather kernels. Each copy moves 472320 bytes. See memcpy_verification.json for the checked timeline.",
        "",
        "## Charts",
        "",
        "The three main figures show random placement. Head and tail figures are in their corresponding subdirectories. PNG, SVG, and PDF versions are available.",
        "",
        "![10% Host](host_10.png)",
        "",
        "![15% Host](host_15.png)",
        "",
        "![20% Host](host_20.png)",
        "",
        "## GPU cycle latency (µs)",
        "",
    ]
    details = [
        "# Component and single-step latency",
        "",
        "All table entries are medians in µs. Cold single: one serialized GPU cycle after 64 MiB L2 eviction. Serial wall: one cycle measured on CPU, including CPU packing for Memcpy, graph launch, and final synchronization. Packing: CPU gather into a pinned transfer buffer. Index updates remain excluded.",
        "",
    ]
    for layout in LAYOUTS:
        cases = cases_by_layout[layout]
        lines += [f"### {layout.title()}", ""]
        for context, label in zip(CONTEXTS, TICKS, strict=True):
            lines += (
                [f"#### {label}", ""]
                + table(cases, context, fractions=FRACTIONS + [0.6, 0.8, 1.0])
                + [""]
            )
        details += (
            [f"## {layout.title()}", "", "### CPU packing", ""]
            + packing_table(cases)
            + [""]
        )
        for metric, name in [
            ("cold_single", "Cold single"),
            ("serial_wall", "Serial wall"),
        ]:
            details += [f"### {name}", ""]
            for context, label in zip(CONTEXTS, TICKS, strict=True):
                details += [f"#### {label}", ""] + table(cases, context, metric) + [""]
    lines += [
        "## Files",
        "",
        "- [Component and serial-step tables](details.md)",
        "- [Exploratory variants and legacy baseline tables](history.md)",
        "- [CSV data](comparison.csv)",
        "",
        "The old baseline artifacts are preserved. The new benchmark source is ../bench_glm52_host_comparison.py; raw JSON retains samples, source hashes, the Git commit, correctness errors, and GPU snapshots. GPU 1 shared the machine with unrelated workloads on other GPUs. This comparison does not measure a complete GLM model.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_comparison.py",
        "MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_comparison.py",
        "```",
        "",
        "Existing raw outputs are protected from overwrite. Choose new output names for a repeat run.",
    ]
    (FOLDER / "report.md").write_text("\n".join(lines) + "\n")
    (FOLDER / "details.md").write_text("\n".join(details) + "\n")
    fieldnames = [
        "layout",
        "context_tokens",
        "host_percent",
        "host_rows",
        "method",
        "gpu_us",
        "p10_us",
        "p90_us",
        "cold_us",
        "wall_us",
        "cpu_pack_us",
        "max_abs_error",
    ]
    with (FOLDER / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for layout, cases in cases_by_layout.items():
            for (context, fraction), case in sorted(cases.items()):
                for mode, label, _ in METHODS:
                    result = case[mode]
                    writer.writerow(
                        dict(
                            layout=layout,
                            context_tokens=context,
                            host_percent=fraction * 100,
                            host_rows=case["host_rows"],
                            method=label,
                            gpu_us=result["gpu_cycle"]["median_us"],
                            p10_us=result["gpu_cycle"]["p10_us"],
                            p90_us=result["gpu_cycle"]["p90_us"],
                            cold_us=result["cold_single"]["median_us"],
                            wall_us=result["serial_wall"]["median_us"],
                            cpu_pack_us=case["cpu_pack"]["median_us"]
                            if mode == "copy"
                            else 0,
                            max_abs_error=case["correctness_max_abs_error"][mode],
                        )
                    )
    history()
    print(f"Tables and figures: {FOLDER}")


if __name__ == "__main__":
    main()
