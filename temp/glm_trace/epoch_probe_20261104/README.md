# Periodic epoch coordinator diagnostic — 2026-11-04

Explicit user-authorized protocol validation, NOT installed in any attention kernel. GPU3/NUMA1, stable10second idle, shared gpu3 lock, start/end noforeign; state records actual source/binarySHA. One dedicated broker CTA plus24/108 producerCTA, each8producerwarps, two outstanding4608B requests/warp,128rounds. Shared73728B/CTA, allows all blocks resident on108SM; no FlashInfer/production architecture changes. Dedicated coordinator cost is included in event and occupancy.

No global CAS/lock/atomicAdd/atomicMax. Per-request mailbox publication/acquire reply, only broker writes virtual clock. Every500/2000ns target epoch a broker warp scans currently issued/published mailboxes, warp32 maxplus prefix, end=max(previous_end,actual_issue)+ceil(bytes/100GB). Only actual requests are charged; no saved idle credits, future reservation or static perCTA share. Consumers local nanosleep64+globaltimer checks, latency added after bandwidth service. Epoch target is minimum cadence, not promise actual service interval; mailbox scan itself can exceed period. Serialization within batch is ascending mailbox index, not hardware arbitration/fairness.

Important limitation: batching by periodic time does NOT itself remove per-request mailbox scanning/publication. Thus this is an experiment in cadence/coordination ownership, not a fundamentally free bandwidth model. Existing requester-flat-combine differs by no dedicated CTA and request-driven broker acquisition. This probe uses dedicatedCTA solely to isolate mechanism; putting it in production would need architecture/resource discussion.

24 traced cases: CTA24/108 ×epoch500/2000ns ×lat0/20us ×pattern dense/one-third delayed100us/all idle100us midlaunch.1622016requests total, all full realpayload exact, charged request count correct, every consume>=deadline, every deadline>=issue+cost+lat, sorted exact maxplus recurrence; bad0. Requests4608B round up47ns vs46.08ns (2.0% conservative); cannot explain allshortfall. Does not cover mixed variable packet bytes or native sparsefar/local stages: delayed/idle means request sparsity only. No mathematical MLA/LRU integration.

Results (GB/s, epoch500ns, dense):

| producerCTA |0us traced|20us traced|0us no_trace|20us no_trace|
|---|---:|---:|---:|---:|
|24|91.097|60.072|91.022|61.543|
|108|88.568|88.474|88.411|88.395|

2000ns epoch similar0us91.474/88.348,20us57.570/88.395. No_trace payload and requestcount checked; no_trace bad0 field does NOT mean runtime recurrence was checked, that is only tracedrun. Tracing overhead small on these points; no systematic multi-pass variation measured.

108producer0us about65sweeps over5.75ms (~88.5us/sweep average), much longer than target500ns/2us; includes phase wait/overlap/end boundaries, not exact active scan duration.24producer similar65sweeps in1.24ms. This strongly limits intended short-window behavior.20us24producer outstanding payload192*2*4608=1.77MB <100GB*20us=2MB, so loss may mix insufficient in-flight+control overhead; not allbrokeroverhead.108producer two-stagepayload~7.96MB avoids that nominal shortage but still~88GB, not target100. No causal per-stagehardwaretrace.

Conclusion: protocol correctness successful for statedcases, no stable100GB performance adoption signal, do not integrate or call nativefullkernelprediction. Better than perpacketCAS pathology does not establish hardware fidelity. Retain negative data, no parameter tuning to forcefullattention result.

First compile exceeded default48KiB staticshared; changed to73728 dynamicshared+attribute, compiled successfully. No failedGPU run. Sources probe.cu/no_trace.cu, raw probe_raw.json/no_trace_raw.json, runners and states. Do not rerun completedrunner filenames; opens x. Binaryignored/rebuild nvcc-O3-archsm_80. Production host81093/math34af/FAR8507 untouched. No suite/sanitizer/physical/H800 conclusion, allrunners ended.
