# Warp-per-row fetch, 2026-11-01

Rejected; production remains b6eea095…e72a / 0f82e9de. Goal remains active. See [report](../../../docs/mla_host_a100_fetch_warp_20261101.md), decisions.json and performance_audit.json.

warp_rows.cuh (a1566bf9…404e) assigns each warp one 1152B row and reuses its page/victim base across 72 vectors for remote issue, staging store and cache store. Only cached whole CTA32 with at least 4×SM rows dispatches the new type. Attention, 64-row double buffers, flags, fences, LRU, deferred copies, far-byte hooks and canonical math remain unchanged. The final 8-vector warp iteration has only 8/32 useful lanes.

105 normal tests passed. All original 34 EMU / 68 normal library kernel instances exactly match in instructions/resources; new kernels REG244, STACK0, SHARED2304, LOCAL0. The actual normal Graph has one WarpRowsHostBatchParams kernel, 108 CTAs × 256 threads, dynamic shared 147968B and one resident CTA/SM. Static instructions decrease 7888→7744, without a runtime gain.

GPU3 three same-parameter pairs at MTP8/R128/LRU4096, 256 positions × 3 cold passes, f24/off: 100GB/s 0µs -0.764%, 20µs -0.030%, 50GB/s 20µs -0.088%. All 6 cells / 1152 timed launches / 72 segments and both references are audited, including all negative samples, misses, guards and source hashes. No full-window/native-link/cross-shape confirmation was attempted. Far source here is separate same-GPU HBM, not NVLink.

make_warp_rows.py records the original transformation. restore_includes.py verifies existing private includes or restores missing ones from frozen source snapshots without overwriting. Source files, CPU builds, tests, actual-dispatch probe, short-screen launcher, audit and raw data are retained. Baseline compiler/screen libraries reuse W/jit_cached_ids (EMU) and W/jit_retained_normal (normal), both exact retained b6; restore/build those W sources first on a fresh checkout. All runners are terminal; see continuation_state.json.

The 00:29:43 +08 read-only H800 query is h800_queue.txt. Five original Slurm jobs remain pending and both watchers remain alive; no queued source or submission changed.
