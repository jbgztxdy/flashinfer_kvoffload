# Frozen attention half-ring experiment

Production remains JQ `7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469`. The direct experiment baseline is pending CPE `a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15`, not the production header.

- `half_ring.cuh`: five physical 16-row attention halves, two logical 32-KV tiles plus one early half. Same single cooperative kernel, exact LRU/cache payload and unchanged simulator.
- `no_copy_pe.cuh`: original smaller compute storage accidentally selects **one** fetch buffer via the existing `sizeof(SharedStorage)` condition. This is a safe specialization, not the out-of-bounds issue originally suspected.
- `no_copy_pe_fixed.cuh`: union padding restores original 147,968-byte storage and **two** fetch buffers. An allocation-only host fix would not change that compile-time policy.
- `invalid_budget/`: historical directory name for the original nine-row cohort. Its initial bounds hypothesis is explicitly retracted. Raw bytes and all positive/negative results are preserved; use `archived_policy_audit.json` for their current paths. Archived Python files are original text snapshots, not directly runnable from that subdirectory.
- `performance_audit.json`: the fresh fixed-control nine-row cohort only. The first cohort is separately audited, not discarded. Three variants each have 105+6 normal tests; the half-ring tests are reused across cohorts and counted once: 333 unique tests.

`restore_includes.py` recreates missing include trees from the frozen headers and refuses mismatched trees. JIT trees and `.so` files are excluded from Git; codegen summaries bind their hashes, full SASS/resource archives, and every original instruction/control word. The `make_*` generators preserve frozen header bytes. Manifests describe measurement-time state, while `decision.json` and the report describe the current decision.

Builds use Python `/mnt/data/mhy-vllm-venv/bin/python`, CUDA13.1, `FLASHINFER_CUDA_ARCH_LIST=8.0`, empty `CUDA_VISIBLE_DEVICES` for CPU-only compilation, `TUNE_VARIANT` selecting a frozen include tree, and a separate `FLASHINFER_WORKSPACE_BASE=.../jit_VARIANT[_normal]`. EMU builds add `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU`; normal BF16/FP16 builds use an empty value. `compile.py` contains the exact module build calls.

`test_candidate.py` and `stress_candidate.py` acquire the shared GPU3 correctness lock and verify resources before/after. `validate_fixed.py` records exact commands/PIDs/exit codes for codegen, actual Graph, all sanitizer gates, and the fixed short screen. Its original process must finish before considering any restart; do not rerun a completed producer just to inspect results. The first memcheck exit86 and its CUDA API-query errors are preserved. Device checks were repeated with `--report-api-errors no`; matched-kernel scope is recorded, not claimed as exhaustive validation.

To inspect completed evidence without scheduling GPU work, run `audit_archived_policy.py`, `audit_performance.py`, `checkpoint_audit.py`, and `report.py` from this directory. The checkpoint expects the built libraries to be present when checking their hashes. Short screen: GPU3 A100-PCIe/NUMA1, same-card separate HBM for EMU far payload, MTP8/R128/LRU4096/f24/off, base→fixed-control→half-ring, 256 positions×3 cold passes, g100_l0/g100_l20000/g50_l20000. This is not a full trace, physical NVLink test, or adoption proof.

`next_mechanism_review.json` is a CPU-only, unimplemented QK/PV warp-role resource review. It is distinct from the older rejected loader/compute warp-role implementation and has no performance evidence. Parent CPE's original SXM producer is independent and remains unchanged.
