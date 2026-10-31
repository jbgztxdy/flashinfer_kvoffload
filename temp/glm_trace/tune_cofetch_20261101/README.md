# CTA-integrated fetch and one-stage control, 2026-11-01

Both rejected. Production remains retained b6eea095…e72a / 0f82e9de; goal active. See [report](../../../docs/mla_host_a100_cofetch_20261101.md), decisions.json and performance_audit.json.

cofetch.cuh (af61098d…f6f) uses one attention stage and a separate 2x16-row buffer, with all CTAs advancing disjoint cyclic fetch queues each eighth KV tile and during tile-ready waits. All pools finish original LRU selection before a uniform grid barrier; every CTA drains its queue before exit. stage1_control.cuh (020b5656…0b41) isolates the one-stage attention change while padding dynamic shared to147968B so the original dedicated64-row fetch double buffers remain intact. Canonical math, emulator, reference and original5304-cell matrix stay unchanged.

Each variant passed105 original normal tests plus2 long-queue/Graph/eviction/payload tests (128pools>108SM, MTP8/K2048, capacities512/4096, all filled cache payloads). All34EMU/68normal original kernel instances exactly match. Actual Graphs each have one newkernel,108CTA x256threads,1residentCTA/SM. Cofetch usesREG244EMU/242normal,staticshared38272,dynamic111104; controlREG240/static2304/dynamic147968. BothSTACK0/LOCAL0.

GPU3 short screen is9cells/6pairs,256positions x3coldpasses,MTP8/R128/LRU4096,f24/off. Cofetch slows100GB0/20us38.940%/53.030%,50GB20us15.761%; control slows11.794%/12.284%/0.470%. No full-window/native-link/cross-shape expansion. All1728timed launches/108segments, two references, source hashes, misses, guards and negative samples are retained. Same-GPU far HBM here is not NVLink.

Original8131 failed control global preflight when another user'scounter entered. Failure/log/state preserved; only missingvariants resumed after verifiedterminal1, both completed. The late read-only GPUmonitor recorded89 one-second samples and0sampled conflicts; it doesnotcover earlierbase/cofetch and doesnotprove sub-second exclusivity. Per-cell pre/end exclusive checks remain. No foreign process was signaled. resume_screen_initial.py freezes the executed launcher; the final launcher fixes byte-offset handling and clarifies foreign-resource-check classification for future reuse.

restore_includes.py validates existingtrees or restores missingtrees from frozenhost/math without overwrite. Baselines reuse exact retained-b6 W/jit_cached_ids and W/jit_retained_normal; restore/build W on a freshcheckout first. All local runners in this directory areterminal. A separate hybrid successor preserves twoattentionstages and dedicated64-row transfers, with small2x4-row worker assistance; its evidence is independent.

H80000:45:46+08 queue snapshot is h800_queue.txt: fiveoriginaljobs stillpending, bothwatchers alive. Remote context was read before the query; no remote mutation/submission.
