# Feature-partitioned QK with two-stage 16-KV tiles, 2026-11-01

Both candidates rejected. Production remains retained b6eea095…e72a / 0f82e9de; goal active. See [report](../../../docs/mla_host_a100_feature_qk_20261101.md), decisions.json and performance_audit.json.

dim_split uses two 32-head items, 128 threads, two 16-row KV stages and a 4096B partial-score exchange. Warp groups cover disjoint PE/CKV features, then sum scores and use original nonsharded softmax/PV. The equation is unchanged but FP32 dot reduction order differs. duplicate_control has the same layout/padding and native duplicated QK. Both only change existing large cached whole-row SM80 dispatch.

Each passed 105 normal tests plus four BF16/FP16 long-sequence/cache-payload cases. All original 34 EMU / 68 normal library instances exactly match, including pure device. Actual BF16 Graphs prove one corresponding kernel, 216 CTAs x128 threads, dynamic78080B / static2176B and two resident CTAs/SM; demangled feature booleans are verified. New REG231 versus control255, both STACK0/LOCAL0. The private e5968 math header only reuses three established 32-head Q-load index changes; benchmark provenance captures both executed private host and math headers. The auditor requires the complete old-instance equivalence proof before reusing canonical-math references.

All 18 cells /12 pairs, 3456 timed launches /216 segments and two references are complete: two short cohorts at f24 and an explicitly planned f48 in-flight-payload control (32-row fetch blocks versus native64). f24 dim_split slows100GB0/20us9.605%/27.923%; f48 still slows same-parameter base2.408%/3.276%, and does not replace retained f24. All positive50GB and negative cells remain. No full-window/native-link/cross-shape expansion.

restore_includes.py validates existing private trees or restores missing host/math pairs without overwrite. Baselines reuse exact b6 W/jit_cached_ids and W/jit_retained_normal; restore/build W first on a fresh checkout. Source, codegen, actual-dispatch, tests, raw data, report generator and audits are retained. The initial generator assertion failure is preserved separately and produced no candidate artifacts or GPU results. All runners here are terminal.

H800 queue/context-hash snapshots are read-only. Five original jobs remain pending and both watchers live; no queued source or submission changed. The next independent unit-rescale experiment has its own directory and gates.
