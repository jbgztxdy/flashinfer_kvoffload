# QPE fragments cached across an attention item

The base is frozen JQ a2f37fb9, not production b6. `manifest.json` binds the
candidate and canonical math headers. Only the existing large cached fully
staged JQ dispatch uses the new derived traits/params and single cooperative
kernel. All original JQ instances remain available for strict codegen comparison.

The new helper loads16 packed QPE words per lane once per item, in the exact
MMA operand layout produced by native ldmatrix.x4. Paired and odd KV tiles reuse
them. PE→CKV accumulation, joint max/rescale/PV/refill, 2stage/32KV/64heads,
256threads, f24 and LRU/cache payloads remain the same. Original shared QPE
allocation and initial copy are retained to isolate reuse. Register pressure or
spill may offset the removed repeated QPE LDSM operations; no gain is assumed.

This differs from N9's per-tile global QPE reads and288thread service-warp
architecture. This experiment retains the original dedicated fetch CTAs and
does not sweep unroll, tile, grid or fetch settings.

Original handles:

- 95227: normal BF16/FP16 CPU compile and105 correctness tests on exclusive GPU3.
- 36824: emulated BF16 CPU compile.
- 61566: wait for normal tests, then6 stress tests.
- 49401: wait for original producers, compare36/72 old kernel instances and
  machine/control words, prove actual single-kernel Graph dispatch, run
  synccheck/racecheck, then fixed JQ→cached_pe large-batch screen and audit.

Do not duplicate live handles or restart after a transient observation timeout.
`validation_state.json` records each child PID and terminal result.

Stress tests cover BF16/FP16, K2048 plus63-tile odd tail,128pools>108SM,
capacities512/4096, empty/single/padding, output/LSE, eviction/zero-miss/Graph,
every filled cache payload, and largeK65 with padded QPE head stride80.
Frozen JQ control tests/binaries are reused and not counted as new tests.

Fixed initial performance plan: MTP8/R128, LRU4096/request, f24/split off,
256 positions×3cold passes, JQ(base)→cached_pe,
100GB0µs /100GB20µs /50GB20µs. GPU3 is A100-PCIE NUMA1; same-card independent
HBM is not NVLink. Shared lock and before/after resource checks are mandatory.

`make_candidate.py` creates/verifies measured headers and missing private include
trees; tracked test/probe/orchestration scripts are authoritative and preserved.
`restore_includes.py` restores missing trees without overwriting existing trees.
`decisions.json` and the generated report state whether any extension or adoption
is supported by the actual evidence. Overall goal remains active.

The initial generator assertion failed before creating candidate/include/JIT,
because the two function signatures had different whitespace. The source and
failure description are preserved under `failures/`. Before any compile, source
inspection then caught constness incompatible with the canonical MMA operand
signature; the uncompiled28d149 header/manifest and correction are also archived.
Only the corrected source in `manifest.json` is eligible for measured results.

Original JQ72075/1398 confirmation/retention processes are independent. Inspect
them first; do not manually install or duplicate their work.

Checkpoint: production JQ retained source7193255e completed its original
72075/1398 processes with fresh root109tests/72codegen/actualGraph validation.
This QPE candidate remains uninstalled. GPU3 has completed76matrixcells plus
short/full/reversed/native confirmations and both unchanged-path brackets:
104cells/49pairs,175520timed launches/11024segments, all references present.
Full100GB speedup versus JQ is0.598%–1.471%; nativeHBM1.315%, PCIe0.019%.
The two original>1% unchanged HBM negatives remain; bracket reductions versus
before/after means are+0.650%(R1) and-0.475%(R256), both within1%.

All original handles95227/36824/61566/49401/66271/66530/62457/40637 are
terminal exit0. The only active experiment handle is15217, parentPID2084930,
childscreenPID2104175, currently waiting for new externalGPU0/1 users
2101252/2101253. It runs16SXM full confirmation cells, gates them, then80SXM
matrix cells, refreshes audit/report and stops without installing. Do not
restart it. GPU3 independent HBM cannot satisfy actualNVLink/MTP1R512 gates.

CPU-only prepare-overlap inspection found only1.35%–1.73% duplicate references
inside simple interleaved warps; no traversal kernel was implemented from that
idea. `followup_resource_review.json` contains an unimplemented half-buffer
attention resource idea with explicit unresolved P layout/register/lifetime
requirements. Neither analysis is a kernel performance result.
