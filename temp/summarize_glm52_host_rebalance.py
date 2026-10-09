# Purpose: Export paired CTA balancing measurements and their validation manifest.
# Usage: python temp/summarize_glm52_host_rebalance.py
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path("temp/host_attention_rebalance")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency(case, method):
    return case[method]["gpu_cycle"]["median_us"]


def main():
    datasets = {
        path.name: json.loads(path.read_text())
        for path in sorted(ROOT.glob("*.json"))
        if "results" in json.loads(path.read_text())
    }
    primary = datasets["full_adaptive_cg.json"]
    baseline = Path("temp/host_attention_balance/baseline_segmented.so")
    baseline_hash = digest(baseline)
    rows = []
    maximum_error = 0.0
    checks = 0
    report = [
        "# Host CTA balancing experiments",
        "",
        "GPU 2: A100-SXM4-40GB, 108 SMs. Host NUMA node 1. BF16, one query, "
        "64 heads, latent 512 + RoPE 64, page size 1, top-k 2048, random placement.",
        "",
        "Each candidate is paired with the frozen segmented binary using identical "
        "inputs and physical KV allocations. Graphs contain 100 serialized cycles; "
        "eight index sets are rotated. GPU times include attention and, for Memcpy, "
        "an actual H2D transfer every cycle. CPU packing, top-k, planning, and index "
        "updates are excluded. Timings are in us. Positive paired changes mean slower execution.",
        "",
        "Phase traces use coarse globaltimer samples and are excluded from latency comparisons. "
        "The early host_read_end_us field includes segmented QK; the corrected field is "
        "host_qk_ready_us. Neither is a pure PCIe transfer measurement.",
        "",
        "## Default policy",
        "",
        "Keep 16 local, 32 Host, and 32 active merge CTAs. At least 410 Host rows "
        "enable ready-first merging in batches of 16; lower counts use bulk merging. "
        "The count is published by local split zero and read after acquiring its "
        "ready flag. Experimental allocations and mixed workers remain opt-in.",
        "",
        "## Primary comparison",
    ]
    for result in primary["results"]:
        report += [
            "",
            f"### Context {result['context_tokens']}",
            "",
            "| Host hits | Memcpy | Direct | Frozen | Adaptive | Paired change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for case in result["cases"]:
            reference, candidate = latency(case, "previous"), latency(case, "overlap")
            report.append(
                f"| {case['host_fraction_requested']:.0%} | {latency(case, 'copy'):.3f} | "
                f"{latency(case, 'direct'):.3f} | {reference:.3f} | {candidate:.3f} | "
                f"{100 * (candidate / reference - 1):+.2f}% |"
            )
    report += [
        "",
        "## All paired trials",
        "",
        "Earlier source snapshots are retained as zero-context patches against commit 8e06b73c "
        "(apply with git apply --unidiff-zero). "
        "The primary source is the current header; each raw JSON records its source hashes. "
        "The frozen binary remains local and its identity is included in verification.json.",
    ]
    for name, data in datasets.items():
        assert data["baseline_module_sha256"] == baseline_hash
        report += [
            "",
            f"### [{name}]({name})",
            "",
            f"Rounds: {data['configuration']['rounds']}.",
            "",
            "| Context | Host hits | Memcpy | Direct | Frozen | Candidate | Paired change |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for result in data["results"]:
            for case in result["cases"]:
                reference, candidate = (
                    latency(case, "previous"),
                    latency(case, "overlap"),
                )
                change = 100 * (candidate / reference - 1)
                report.append(
                    f"| {result['context_tokens']} | {case['host_fraction_requested']:.0%} | "
                    f"{latency(case, 'copy'):.3f} | {latency(case, 'direct'):.3f} | "
                    f"{reference:.3f} | {candidate:.3f} | {change:+.2f}% |"
                )
                maximum_error = max(
                    maximum_error, *case["correctness_max_abs_error"].values()
                )
                checks += len(case["correctness_max_abs_error"])
                row = {
                    "trial": name,
                    "rounds": data["configuration"]["rounds"],
                    "context": result["context_tokens"],
                    "host_fraction": case["host_fraction_requested"],
                    "host_rows": case["host_rows"],
                    "host_bytes": case["host_bytes"],
                    "paired_change_percent": change,
                }
                for method in ("copy", "direct", "previous", "overlap"):
                    for statistic in ("median_us", "p10_us", "p90_us"):
                        row[f"{method}_{statistic}"] = case[method]["gpu_cycle"][
                            statistic
                        ]
                rows.append(row)
    with (ROOT / "all_results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "report.md").write_text("\n".join(report) + "\n")
    correctness = {
        path.stem: json.loads(path.read_text())
        for path in sorted(ROOT.glob("correctness*.log"))
    }
    assert all(check["passed"] for check in correctness.values())
    sanitizers = {}
    for path in sorted(ROOT.glob("*check*.log")):
        if path.name.startswith(("racecheck", "synccheck")):
            summary = re.findall(r"========= .*SUMMARY: .*", path.read_text())[-1]
            assert "0 errors" in summary, summary
            sanitizers[path.stem] = summary
    source_paths = (
        "include/flashinfer/attention/mla.cuh",
        "temp/bench_glm52_host_comparison.py",
        "temp/test_glm52_host_overlap.py",
        "temp/profile_glm52_host_balance.py",
    )
    manifest = {
        "primary": "full_adaptive_cg.json",
        "default_ctas": {"local": 16, "host": 32, "merge": 32},
        "adaptive_host_merge": {
            "minimum_rows": 410,
            "batch": 16,
            "metadata_slot": 3500,
        },
        "baseline": {
            "source_commit": "da4ede5f507b50af6078ac5f0887bc5d773c031f",
            "module_sha256": baseline_hash,
        },
        "source_sha256": {name: digest(Path(name)) for name in source_paths},
        "primary_matches_current_source": all(
            primary["source_sha256"][name] == digest(Path(name))
            for name in source_paths[:2]
        ),
        "paired_fp32_checks": checks,
        "maximum_output_absolute_error": maximum_error,
        "correctness": correctness,
        "default_policy_validation": "correctness_adaptive_cg",
        "correctness_cli_note": "Flags in test logs describe explicit overrides. The default invocation uses the adaptive policy from the header.",
        "sanitizers": sanitizers,
        "raw_data_sha256": {name: digest(ROOT / name) for name in datasets},
        "source_archives_sha256": {
            path.name: digest(path) for path in sorted(ROOT.glob("*.patch"))
        },
    }
    assert manifest["primary_matches_current_source"]
    (ROOT / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(datasets)} trials, {len(rows)} cases, {checks} FP32 checks.")


if __name__ == "__main__":
    main()
