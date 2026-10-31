# Paired QK for two different KV tiles, 2026-10-31 / 11-01

Both schedules rejected. Production remains b6eea095…e72a / commit 0f82e9de. Goal remains active; see [report](../../../docs/mla_host_a100_qk_pair_20261101.md) and decisions.json.

qk_pair.cuh (75ea7044…d6f9f) shares each Q fragment across two distinct KV tiles with independent scores. Each score preserves PE-then-CKV MMA order and the loop preserves descending softmax/PV order. It only changes existing cached whole CTA32 instances with at least 4×SM query rows. Fetch, cache, emulator, canonical math header and the single-kernel contract remain unchanged. Historical regpipe3 added a third fetch register buffer, not this attention schedule.

qk_pair_prefetch.cuh (adc6aef1…f155fb) independently restores early next-page-ID reads before paired QK/PV. Its predecessor and private include/JIT were not overwritten. Both schedules wait for the two stages at pair entry, unlike the original rolling one-tile loop. Do not attribute all performance differences to Q reuse alone.

Each candidate passed 105 normal CUDA tests. Actual normal-build graphs prove one new PairedQKHostBatchParams kernel, 108 CTAs × 256 threads, 147968 dynamic shared bytes and 1 resident CTA/SM. All original 34 EMU / 68 normal library kernel instances have exactly equal instructions and resources. New kernels use REG255; first schedule STACK0, early-ID schedule STACK8. Both LOCAL0; the second is not a no-spill result.

GPU3 completed 12 cells / 6 same-parameter pairs: MTP8/R128/LRU4096, 256 positions × 3 cold passes, fetch24/off, 100GB/s at 0/20µs and 50GB/s at 20µs. First schedule is 8.1–8.5% slower at 100GB/s. Early-ID schedule reductions are -0.130%, +0.115%, +0.067%; no stable adoption signal. All samples, misses, references, source hashes and negative cells are retained by performance_audit.json (2304 timed launches, 144 accepted segments, no pending references).

The first base attempt failed exclusive-GPU preflight when another user's counter_payload entered; append-only log and state preserve the failure and successful base-only retry. SXM remained occupied by other users, so only our still-waiting runner was canceled after rejection. sxm_wait_cancellation.json records that no SXM launch was started. No full-window, normal physical-link or cross-shape performance confirmation was attempted. GPU3 emulated far reads use independent same-GPU HBM; they are not NVLink.

restore_includes.py verifies existing private trees or rebuilds missing ones from frozen host and canonical math snapshots; it never overwrites an existing tree. CPU compilers, test launcher, actual-dispatch probes, screen launcher and performance auditor are retained. Baseline compiler/screen workspaces reuse the exact retained-b6 libraries in W: jit_cached_ids for EMU, jit_retained_normal for normal CUDA. On a fresh checkout first restore/build those W libraries using its frozen cached_ids source. screen_initial.py archives the parser before the prefetch variant was added; kernel and benchmark sources remained frozen while the first SXM runner was waiting.

All runners in this directory are terminal. continuation_state.json records their final handles and the next independent fetch-row experiment.
