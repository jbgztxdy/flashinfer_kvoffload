# Purpose: Export paired CTA allocation results and critical-path evidence.
# Usage: python temp/summarize_glm52_host_balance.py
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent / "host_attention_balance"


def load(name):
    return json.loads((FOLDER / name).read_text())


def cases(result):
    return {case["host_fraction_requested"]: case for case in result["cases"]}


def comparison(case):
    reference = case["previous"]["gpu_cycle"]["median_us"]
    candidate = case["overlap"]["gpu_cycle"]["median_us"]
    return reference, candidate, (candidate / reference - 1) * 100


def main():
    verification = load("verification.json")
    identity = verification["baseline"]
    lines = [
        "# CTA allocation experiment",
        "",
        "## Result",
        "",
        "Redistributing CTA work did not improve 25% or 30% Host hits in the tested configurations. The default remains 16 local and 32 Host compute partitions. Automatic allocation switching is not enabled.",
        "",
        "The 44-local/32-Host configuration uses all 108 CTAs: 76 compute partitions and 32 merge CTAs. Across 128K-1M contexts, its 25% latency increases by 3.7%-4.9%, and its 30% latency increases by 4.0%-5.0%, against the paired frozen segmented reference.",
        "",
        "## Complete paired comparison",
        "",
        "Random placement, BF16, one decode query, 64 heads, top-k 2048. Forty rounds, eight index sets, 100 serialized cycles per graph. Times are median GPU cycles in us. Reference and candidate share identical inputs and physical KV allocations within each run; method order is randomized.",
        "",
        "| Context | Reference 25% | 44/32 25% | Change | Reference 30% | 44/32 30% | Change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in load("full_local44_host32_random.json")["results"]:
        selected = cases(result)
        values = []
        for fraction in (0.25, 0.3):
            reference, candidate, change = comparison(selected[fraction])
            values.extend([f"{reference:.2f}", f"{candidate:.2f}", f"{change:+.1f}%"])
        label = {131072: "128K", 262144: "256K", 524288: "512K", 1048576: "1M"}[
            result["context_tokens"]
        ]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Additional allocation candidates",
        "",
        "Each entry uses its own paired reference. Positive changes mean slower execution. CTAs are listed as local/Host compute partitions.",
        "",
        "| Allocation | Merge reads | Rounds | 25% change | 30% change | Artifact |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, label, merge in [
        ("fixed_local24_host32_random.json", "24/32", "Bulk"),
        ("full_local32_host32_random.json", "32/32", "Bulk"),
        ("full_local44_host32_random.json", "44/32", "Bulk"),
        ("fixed_local16_host48_random.json", "16/48", "Bulk"),
        ("full_stream_local44_host32_random.json", "44/32", "8 partitions per batch"),
    ]:
        data = load(name)
        result = next(
            row for row in data["results"] if row["context_tokens"] == 1048576
        )
        selected = cases(result)
        changes = [comparison(selected[fraction])[2] for fraction in (0.25, 0.3)]
        lines.append(
            f"| {label} | {merge} | {data['configuration']['rounds']} | {changes[0]:+.1f}% | {changes[1]:+.1f}% | [{Path(name).stem}]({name}) |"
        )
    data = load("full_local32_host32_random.json")
    selected = cases(data["results"][0])
    low_hits = ["", "The 32/32 allocation improves 0%-15% in its 1M paired run:"]
    for fraction in (0, 0.1, 0.15, 0.2):
        reference, candidate, change = comparison(selected[fraction])
        low_hits.append(
            f"- {fraction:.0%}: {reference:.2f} to {candidate:.2f} us ({change:+.1f}%)."
        )
    lines += (
        low_hits[:2]
        + [""]
        + low_hits[2:]
        + [
            "",
            "These lower-hit-rate gains do not establish a benefit at 25%-30%.",
            "",
            "## Critical path evidence",
            "",
        "Globaltimer traces come from a separately instrumented 16/32 kernel, measured from the earliest CTA entry. Times below are in us. They are diagnostic single-call medians, have coarse timer granularity, and are excluded from the latency comparison. Individual medians need not add up.",
            "",
            "| Host hits | Local compute complete | Host compute complete | Merge complete | Tail after compute |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in load("phases.json")["cases"]:
        times = case["median_us"]
        values = [
            times[key]
            for key in (
                "local_end_us",
                "host_end_us",
                "merge_end_us",
                "post_compute_us",
            )
        ]
        lines.append(
            f"| {case['host_fraction']:.0%} | "
            + " | ".join(f"{value:.1f}" for value in values)
            + " |"
        )
    lines += [
        "",
        "Local compute already completes well before the Host side. Accelerating local compute alone does not shorten the observed critical path. Increasing partitions also writes more partial outputs and increases merging work. The traces do not isolate PCIe saturation or pure computation, so they cannot prove that link bandwidth is the only limit.",
        "",
        "Some larger bulk-merge candidates introduce register spills. Batching merge reads removes spills in the final 44/32 streamed candidate (238 registers, zero stack), but that candidate still regresses 25% and 30%. Further work should shorten Host-dependent reading, computation, and merging rather than target SM occupancy alone.",
        "",
        "## Verification and provenance",
        "",
        "- GPU 2: A100-SXM4-40GB, Host NUMA node 1. GPU 1 had substantial allocations from other processes. All reference/candidate comparisons use the same GPU. Absolute timings vary across separate allocations; use paired changes when comparing trials.",
        f"- Frozen reference: source commit `{identity['source_commit']}`, module SHA256 `{identity['module_sha256']}`. Local binary copy: `baseline_segmented.so`.",
        f"- The generalized 16/32 control matches the frozen binary within {verification['default_control_max_relative_difference'] * 100:.2f}% across six hit rates. Both use 242 registers and zero stack.",
        f"- All {verification['paired_fp32_output_checks']} paired FP32 output checks passed; maximum absolute error {verification['paired_maximum_absolute_error']:.6f}.",
        "- Extended BF16/FP16 outputs, base-2/base-e LSE, random/head/tail placements, graph replay, Host mutations, tile boundaries, and the maximum 84-partition scratch layout passed 420 checks. This includes the four-head merge path.",
        "- Racecheck: zero hazards/errors/warnings. Synccheck: zero errors.",
        "- Original Memcpy/Direct/Segmented reports, figures, and raw datasets remain the baseline measurements. This experiment does not replace them.",
        "- Initial scouting results and register reports are retained. Applying `before_loop_fix.patch` with `git apply --unidiff-zero` to the frozen reference source recreates the initial kernel used by the `stream_*` scouting runs. Its dynamic head loop introduced extra spills. The final compile-time pass loop restores the 16/32 control and is used by the primary measurements.",
        "",
        "[Validation manifest](verification.json) · [Phase samples](phases.json) · [Original comparison](../host_attention_comparison/report.md)",
        "",
        "## Reproduce the complete comparison",
        "",
        "```bash",
        'PYTHONPATH=. FLASHINFER_WORKSPACE_BASE="$PWD/temp/jit_host_balance" \\',
        '  FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR="$PWD/temp" \\',
        "  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=4 \\",
        "  /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_comparison.py \\",
        "    --device 2 --cpu-affinity 32-63,96-127 \\",
        "    --host-kernel private --private-chunks 32 --private-pipeline \\",
        "    --local-chunks 44 --contexts 131072 262144 524288 1048576 \\",
        "    --fractions 0 0.1 0.15 0.2 0.25 0.3 --rounds 40 --index-sets 8 \\",
        "    --baseline-module temp/host_attention_balance/baseline_segmented.so \\",
        "    --output temp/host_attention_balance/new_full_local44_host32_random.json",
        "```",
        "",
    ]
    (FOLDER / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
