# Isolated request batching / local deadline wait — 2026-11-03/04

User explicitly authorized fixing/replacing the emulator, then testing100GB. Production host81093ba5 and FAR8507eab0 remain unchanged. All include/JIT/binaries isolated here. GPU3/NUMA1 same-cardHBM only, notNVLink. GPU0/1/2 and H800 untouched. No agents. Old runners are completed, not resumable; make/prepare scripts are one-time constructors.

## What is already established

Oldv2 smallpacket108CTA/10rows puretransfer reaches31–34GB/s. Actual retry diagnostic: ~53CAS retries/request (max>9000) vs~0.5 for24CTA/64rows. Backoff makesitworse19/14GB/s, rejected; original logs retained. No production change.

`batch_first.cuh`/`batch.cuh`: a requester warp wins one broker lock and flat-combines **already-published, already-real-issued requests**. It performs32request maxplus prefix scan, publishes each request's own deadline; consuming warps locally wait. No future budget, no dedicatedSM, no changing byte charge/reference, no static perCTA bandwidth. Current batch adds GPU acquire/release fence around broker ownership. Mailbox reuse is perCTA/warp, one outstanding reservation/requester; mask cleared before release answer. Fixed priority CTA/warp order is one allowed serialization, not a model of physical arbitration fairness.

For each ordered request: end=max(previous_end,recorded_issue)+ceil(actual_bytes/B); latency added after serialization, not charged to bandwidth. No earlier consumption than end+L. Current model rounds each packet up;10rows cost116ns at100GB vs ideal115.2ns, a known~0.7% conservative error at sustained smallpackets.

`batch_verify` checked realpayload + everyrequest recurrence/charged sum/consume>=deadline for10/64row,100GB,0/20us. `semantic` checked154368requests,2/24/108CTA,1/8activewarps,50/100GB,0/20us,32rounds,mixed1B/1152B/10row/64row, delayedwarp and100usidle. All bad/early0. Independent proof of these workloads, not all fullkernels or physicalNVLink equivalence.

`calibrate` finalfencedbatch no trace:100GB10row97.86/92.15,20row99.26/98.37,32rows perwarp4rows72.82/72.68,64row98.84/96.71GB/s (0/20us);50GB all~49.2–49.9. `batch_result` earlier98.28/94.77smallpacket belongs batch_first, not currentfencedversion; do notmerge.

`batch_cta` explicit fullCTA aggregation: everythread first issues own actualreads, then counts currentstage bytes; warp0 reserves their sum. No future stage bytes. Currentstage consumption already depends allwarps. More conservative issue time is after wholeCTA reachedhook, and CTAbarriers add overhead: this is explicit and must be quantified. `calibrate_cta`:100GB32row99.43/98.63;20row99.22/98.18;10row98/92.08;50GB~49.5–49.9.8warp requests becomeone packet onlywhere callers guaranteefullCTA participation.

`cta_scalar`: same explicit fullCTA aggregate but old exactCAS reservation, simpler code. Puretransfer32row97.66/90.99 at100GB and~49.8/49.6at50; small10/20 stillcontention. This is a diagnostic candidate, not adopted.

## Full MLA integration and limits

Private `staged`: productionkernel+batch. `uniform`: rejected original108CTA10rowunique kernel+batch. `direct`: currentdirectmathenabledinEMUonly, actual32KVstage aggregation withbatch_cta and real perstageFAR waits. Do not enable production direct normal-only branch without these waits. First compile lacked State local declaration, archived failed_direct_compile. `direct_warp_frozen` compiled but neverran onGPU; actualintegrationusesfullCTA stage hook.

All3 got2focusedchecks each:K65/K2048,padding/empty/odd,QPEstride80,output/LSE,Graph×2,3cachetraces,allcachepayloads/mappings. Uniform alsoexactuniqueissue/publishcounters. Baseline uses existing exactv2EMU JIT. Original staged100GB0/20 3288.395/3293.429us; newmodelstaged3307.787/3320.501us (~+0.6/0.8%). Thus oldstagedgap persists undernewmodel, notjustCASartifact. References unchanged.

**Fullkernelcounterexample to assuming calibratedcopy means negligibleintegrationoverhead:** uniforminline REG128/STACK944,100GB~11ms; noinlinebroker REG255/STACK32,0us5172us; original oldv2uniform9.7ms, normal~3.9ms. Directinline REG255/STACK8,0/20~8762/12073us; noinline STACK112~9327/12426us. These are prototype fullkernelmeasuredtimes, not accepted physical100GB predictions. Moreissue/scheduling/control overhead must be investigated; do notsubtract it or silentlyuse results as formalreference.

`read_symbols.py` directly calls cudaMemcpy{From,To}Symbol with verified ELF local hostshadowaddresses, outsideevents. `accounting_direct`: reads actualbroker records on coldstep0 and one transition step31. Exactmaxplus charged actualfarrows==decodedremap references. Cold2.414GB, secondtransition1.292GB. No duplicatebilling detected; transition is **not warmed32steptrace**. Trace disabled before benchmark. Pytorch can switch tracing after builds, modelcontrol only, no GPU mathematicalwork is removed.

`uniform_call`/`direct_call` noninline onlychangebroker callsite, not model semantics. Bothcompiledandfocusedchecked. `direct_scalar` compiled,2focusedchecks pass,1pass100GB0/20 gives19.69/21.51ms, rejected. `control_*` samebinaryruntimeenabledunlimited/disabled isolate generichook overhead. Catalogfailure beforeGPU in failed_control_catalog; skip alreadycompleted nativecontrol on resume. Bench settings metadata can saybandwidth100 forcontrol g0: actual configname/g0 and runtime env in harness determine executedsetting; do notclaim100.

An overlappingowned microprobe took sharedGPUlock justbefore staged20bench, which failedbeforeCUDA withBlockingIOError. Archivedfailed_lock_preCUDA; completedmissingstaged/base20separately withresume_staged. Never runGPUrunners simultaneously, even ifidle snapshotpasses.

## Reproduction / audit

Python /mnt/data/mhy-vllm-venv/bin/python, CUDA13.1 arch8.0, NUMA1, GPU3 stable10s free, sharedgpu3lock. compile.py privateworkspacejit_variant and privateinclude_variant; noGPUbuild. Frozenheaders authoritative, manifest SHA bindsfinal and earlier sources. Includes/JIT/binariesignored, rebuildfromfrozenbase/candidate+canonicalmathcurrent34af. Do not rerun prepared one-timescripts overexistingfiles or completedrun scripts/overwrite raw. Everynewrunner has boundedpreflight/children; no otherusers killed.

Independentcopy baselines nohooks/enabledunlimited/limited, all actualpayloadchecked. FullMLA timing Graph/events,256position32steps,3coldpasses forstaged/uniform;1passdiagnostic forcall/direct. Fullcachepayload and sampledmath checked outsideevents. Oldmatrix/reference/productionFAR untouched. `calibrate_sparse` actual32rowstage with8farrows:100GB0/20=91.61/70.01;50GB=49.24/47.88. This sparse/latency envelope is not solved.

Current results are an isolatedcandidate, not blanketmodelvalidity or 110%goalcompletion.
