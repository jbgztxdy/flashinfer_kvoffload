# Two-stage attention with small worker fetch assistance, 2026-11-01

Both versions rejected; production remains b6eea095…e72a / 0f82e9de, goal active. See [report](../../../docs/mla_host_a100_hybrid_cofetch_20261101.md), decisions.json and performance_audit.json.

hybrid.cuh (9168c83f…8259) preserves two attention stages. Dedicated CTAs fetch from 2x64-row dynamic buffers before joining attention; workers assist from separate 2x4-row static buffers each eight KV tiles and during waits. An atomic cursor assigns disjoint 16-row jobs. Worker flags publish after all four subchunks land; dedicated claims cover four jobs. The gap flag is reused only after prepare/compaction and reset by the existing last-CTA exit. LRU selections finish before issue. Math, payloads, capacities and original emulator hooks remain unchanged.

The first version carries both runtime service roles through attention, with REG255 / STACK440 EMU and STACK432 normal. hybrid_static.cuh (7cac4924…df25) statically separates dedicated 64-row and worker 4-row service while preserving the same policy, jobs, buffers and cadence. It uses REG245 / STACK0 in both builds. This is a compiler-liveness control, not an unroll/cadence sweep. Original source and large-stack evidence were not overwritten.

Each version passed 105 original normal tests plus two long-queue/Graph/eviction/full-cache-payload tests. All 34 EMU / 68 normal original library instances exactly match in instructions and resources. Both actual normal Graphs contain one corresponding new kernel, 108 CTAs x 256 threads, dynamic shared147968B / static11264B and one resident CTA/SM.

Two independent GPU3 cohorts completed 12 cells / 6 pairs, MTP8/R128/LRU4096, 256 positions x 3 cold passes, f24/off. First version slows 100GB/s0/20us by108.520%/107.337%, 50GB/s20us by58.418%. Static version slows13.881%/14.911%/12.855%. Base order is forward for the first cohort and reverse for the second. All 2304 timed launches / 144 accepted segments, both references, negatives, misses, guards, per-step/pass data and source snapshots remain. No full-window/native-link/cross-shape expansion. Far HBM here is separate same-GPU memory, not NVLink.

CPU compilers, test and stress launchers, actual-dispatch probes, screen, performance/checkpoint auditors and restore_includes.py are retained. Restoration validates existing trees or rebuilds missing trees from frozen host/math without overwriting. Baselines reuse exact retained-b6 W/jit_cached_ids and W/jit_retained_normal; restore/build W first on a fresh checkout. All local runners are terminal; continuation_state.json lists their original handles.

H80001:20:24+08 read-only queue snapshot is h800_queue.txt: all five original jobs remain pending and both watchers are alive. No queued source or submission was changed.
