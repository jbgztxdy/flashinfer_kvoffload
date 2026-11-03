# Far-link model v2: correctness fix and calibration

User explicitly authorized fixing the simulator before further kernel optimization.
Old model and historical raw are preserved; see baseline.json and sibling
emu_semantics_20261103. Attention host/math remain cab508dc/34af; no kernel
optimization or reference change is included here.

Fixes:
- CAS atomically updates max(clock,issue)+ceil(bytes*ps_per_byte/1000)ns.
  Idle/busy/mixed reservations cannot lose service. Nanosecond rounding is
  conservative, less than1ns perpacket (0.098%at64rows/100GBps).
- Warp packets reserve at issue, rather than being charged retrospectively
  at consume. Each CTA/stage records its latest deadline with atomicMax.
- A prewait CTA barrier ensures every loader warp published its deadline.
  Existing postwait barriers protect consumption/reuse. Legacy State aggregate
  layout retained for manual initializers; prefetch is now a compatibility no-op.
- No arbitrary1second truncation of valid deadlines. Latency is added to each
  packet's virtual serialization completion, not into the serialization clock.
- Memcpy gather reserves before reads and uses64row packets/48singlebufferCTAs.
  This has3072inflight rows, like f24double64row fetch. Original432x16layout
  was measured at~51GBps with correctCAS; failed calibration remains in
  cas_contention/. It was NOT accepted or used for a new MLA speed claim.
- Publicemu binding uses build_v2/name far_emu_gather_v2 and rejects mismatched
  frozen kernel/gather headers before build/GPUwork. Oldbuild remains untouched.

Validation on exclusiveGPU3/NUMA1,sharedlock:
32reservationcases/16000trials (50/100GBps,0/20us,2/24CTA,idle/busy,mixed),
3arithmeticcases (idle gaps,nonmonotone issue times,zero bytes,unlimited),
2stagecases with deliberately late warps,6realpayload transportcases including
partial13rowtail. No accounting/earlyrelease/payload failures.
Actual24CTA,double64row cp.async transport:100GB0/20us99.53/98.78GBps;
50GB0/20us49.91/49.75GBps. Seeverification.json.
Standalonegather15rate/latencysettings (0/50/100GBps×0/2/5/10/20us):
100GB98.17..99.15GBps,50GB49.69..49.88GBps. Every payloadbytechecked.
Allsamples satisfy prior5%rate and10%latency+600ns criteria; positive20us
chase~20.48us. No guarantee of zero physical-access time whenL=0.
3BF16attentionintegrationtests passed (stagedK65/K2048,directK65,output/LSE,
3cacheepochs,allvalidcachepayload/mappings,Graphreplays), plus publicbinding
payload/Graph2replay check and CPUrejectionofoldheader.

Transport-only overhead diagnostic (~302MB):nohooks486.4us,disabledhooks558.4us,
enabled/unlimitedhooks695.0us. These are microkernel timings, not a causal
MLA time decomposition; do not subtract208.6us from oldMLA results. Newactual
MLA timings must be collected under v2 before discussing the110%gap.

Bounds: payload availability model, not physicalNVLink equivalence. Same-card
far reads consume localHBM/L2; software CAS/barriers/polling have overhead.
Single active launch per module: perCTA slots are shared; separate modules
have separate clocks. Calibrated50/100GBps,0..20us; do not claim arbitrary
bandwidth or concurrentmulti-stream fidelity. Historical results stay v1;
old runners with frozenv1headers cannot be reused with currentgather.

Entry points:prepare.py is one-time;build.py isolatesnormal/EMUincludes/JIT;
run_verify.py,calibrate.py,run_integration.py,test_binding.py record resources;
audit.py checks saved results and provenance. Already-completed producers
must not be restarted merely to inspect evidence. first_attempt preserves
compilecompatibility and test-oracle failures; finalresultsexclude them.
