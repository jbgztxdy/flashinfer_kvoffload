# GLM-5.3 kernel-only matrix

The report is `docs/mla_host_glm53_kernel_matrix_20261029.md`. This directory contains the complete experiment definition, code, per-step data, calibration, failure history, source snapshots, audit, and flat tables. A partial report is explicitly labelled and must not be presented as a completed sweep.

Run from the FlashInfer repository root:

```bash
python3 temp/glm_trace/kernel_report_20261029/run.py
python3 temp/glm_trace/kernel_report_20261029/audit.py
python3 temp/glm_trace/kernel_report_20261029/report.py
python3 temp/glm_trace/kernel_report_20261029/status.py
```

To wait for the original GPU pair and resume interrupted work, run `resume.py`.
It observes GPU0/1 without creating CUDA contexts, waits while other compute
processes occupy either GPU, repeats the recorded unstable no-LRU groups, and
then resumes the full matrix. Its current state is `resume_state.json` and the
owning PID is recorded in `resume.pid`. Check that PID before starting another
controller. Every benchmark checks exclusive GPU use again before accepting data.

`export_html.py` generates the portable HTML report with embedded images and a
filterable data table. It contains the full measured matrix and can export a CSV
for the current filter without a network connection.

`run.py` uses only GPUs 0/1 sequentially, NUMA0, the existing `/mnt/data/mhy-vllm-venv` environment, CUDA 13.1, and distinct normal/emulated JIT directories. It skips complete files, retains incomplete runs under `failures`, and retries guard failures with longer guards. Do not launch another GPU benchmark concurrently.

Completion requires 39 local references, 4,641 no-LRU cases, 663 continuous-LRU cases, 273 exact no-LRU local snapshot cases, 15 calibrated virtual links, and `audit.json: complete=true`. Each comparison reports our fixed batched staged kernel, the actual copy/cache/scatter/native-attention GPU path, and a normal-build all-local attention reference.

LRU: every pass replays all 1,024 positions, cold first step included, three passes. MTP1 has 1,024 launches; MTP8 has 128 launches. No-LRU samples four spaced trace groups for each requested Host fraction. All means are arithmetic. CPU planning, packing, input metadata and checks are excluded from kernel events.

`matrix.csv` and `matrix.json` are derived data. Raw JSON files contain every timed launch, pass mean, correctness anchor, guard, cache miss diagnostic, trace binding, source SHA and VMM residency. Source snapshots are addressed by SHA-256. Failed/preliminary runs never enter the main tables.

The real trace contains four assistant windows and 21 indexer layers per window. Request counts above 84 reuse captures with independent remote/cache addresses; local near data aliases after 128 requests to fit memory. Q/KV payloads are seeded BF16. This evaluates real access patterns, not complete model quality or generated MTP acceptance. See the report for all limits.
