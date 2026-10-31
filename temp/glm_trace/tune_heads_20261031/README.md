# Two resident half-head CTAs, 2026-10-31

Rejected after actual-dispatch proof, 105 normal tests, complete original-kernel codegen equivalence and nine paired short-window screens. Production stays unchanged. [Report](../../../docs/mla_host_a100_half_heads_20261031.md).

`make_half.py` generated frozen private host and math headers. Do not rerun over this directory. The new 32-head/128-thread type writes disjoint heads without a head merge. `dispatch_probe.json` confirms one cooperative launch with 216 CTAs and two resident CTAs/SM; this is not a fallback-only test. `compiler_heads.py` compares all kernels in both ordinary and host-overlap libraries, including pure-device references. Normal64 and emulated32 original library instances are unchanged. New unique symbols1/2, stack/local0.

GPU3 limited-link LRU screens are 256 positions, three cold passes, MTP8/R128, fetch24/32/48 paired against the same parameters. The best candidate settings do not beat the current recommended base fetch24 on100GB/s. No full-window/native/global-retention claim is made. All negative cells, tests, hashes, compiler count failure, source snapshots and raw samples remain. This round's GPU runners are terminal; no candidate installed.

`audit_performance.py` explicitly requires normal original-kernel equivalence before reusing the unchanged same-hardware compute references despite the private math-header SHA change. It retains the original max-per-step reference formula. The older fixed128-CTA diagnostic is never used here. Source snapshots record the executed private host and math headers.
