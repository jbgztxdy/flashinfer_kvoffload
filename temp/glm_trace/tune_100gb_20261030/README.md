# 100 GB/s kernel tuning (A100-SXM4)

The production change is in `include/flashinfer/attention/mla_host_batch.cuh`;
`base.cuh` and `final.cuh` retain the before/after source. Each explored header has
its own `include_<variant>` copy and `jit_<variant>` cache, excluded from git.
The `.cuh` candidate files and executed source snapshots are retained. No link
emulator source was changed.

The new path requires staged reads, `host_split=False`, 64 heads, at least four
query rows per SM, and enough staging for every selected entry. It writes whole
rows directly instead of producing tail partial states and merging them. Existing
small, overflow, in-place, and default split paths retain their original kernels.
`compiler_summary.json` compares the six generic BF16 kernels: instruction text
and resources are unchanged. The measured recommendation for A100 100 GB/s is
fetch24/split off at 432+ query rows and fetch16/split on below that threshold.

The original GLM53 matrix remains unchanged. `bench.py` derives from the audited
full-window benchmark with CLI parameters and isolated include paths. Full
measurement settings, raw per-step samples, source hashes, cache diagnostics and
queue guards are retained in `raw/` and `source_snapshots/`. Independent GPU work
is serialized via the original GPU01 lock, and scripts wait for idle resources.
GPU2/3 and other users' jobs are not modified.

Workflow:

- `phases.py`: debug timestamps from the original kernel. This is diagnostic data,
  not a replacement for the uninstrumented performance matrix.
- `run.py`, `run_stage2.py` ... `run_stage5.py`: exploratory candidates.
- `validate.py`: final 195 full-trace 100 GB/s cells, 33 regression cells on three
  other links, old-code controls, and 168 before/after no-LRU comparison cells.
- `test_targeted.py`: early whole-row Graph/padding/cache-payload tests.
- `test_final.py`: full attention test suite in a normal build.
- `confirm.py`: same-session five-pass default/old-same-parameter/new comparison.
- `report.py`: independently audits samples, guards, means, trace bindings,
  cold-start counts and source hashes, and generates CSV/Markdown.
- `plot.py`: final figure; rebuild the report afterward to include its reference.
- `compiler.py`: compressed SASS and resource/instruction evidence.
- `fetch_scan.py`: a follow-up 20/28 fetch-CTA scan on the same final header and
  benchmark script. These six rows are identified by the `neighbor_` raw filename
  prefix and kept outside the exact 195-cell final matrix. An optional replay with
  a different variant label failed before CUDA allocation because its isolated
  include directory was not created; the original measurements remain intact.

Supervised runs keep `signal*.json`; successful completion is explicitly recorded
there. Failure logs are preserved. Source snapshots capture the exact executed
benchmark even when the driver is subsequently extended.

Final report: `docs/mla_host_a100_100gb_optimization_20261030.md`.
Rebuild derived artifacts after every required run completes:

```bash
python temp/glm_trace/tune_100gb_20261030/report.py
python temp/glm_trace/tune_100gb_20261030/plot.py
python temp/glm_trace/tune_100gb_20261030/report.py
python temp/glm_trace/briefing_report_20261030/build_report.py
python temp/glm_trace/briefing_report_20261030/check_report.py
```

`audit.json: complete=true` requires all 195 trace cells, all 84 final no-LRU
cells, all 12 final five-pass cells, and a successful full test suite. Remaining
reference gaps and slower cells are preserved; a lower elapsed time alone does
not establish that every shape is within 10% of an optimistic overlap reference.
