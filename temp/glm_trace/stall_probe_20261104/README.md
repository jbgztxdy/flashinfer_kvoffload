# Fixed-stall / static-share diagnostic

GPU3 same-device HBM, NUMA1, 2026-11-04. Independent CUDA executable, shared gpu3 lock; no MLA changes. Probe uses real 4096B/CTA/round loads and stores, 512rounds. No payload validation/counters; cudaGetLastError success only, not full correctness evidence. Extra stride288 uint4 while256 copied: useful bytes formula counts only256*16. No production adoption.

Mode0 unthrottled; mode1 thread0 nanosleep4424ns plus CTA barrier; mode2 local globaltimer deadline with4096*108/100ns per request, max(previous_deadline,current_issue), static assumed108 contenders. No global atomics. This static share is explicitly a diagnostic/rejected universal model, not a new accepted bandwidth model or repeated tuning scan.

raw.json: mode1 CTA2/24/108 useful throughput0.999/11.988/53.934GBps; mode2 1.597/19.148/86.232. Static fraction prevents low contender count from borrowing idle bandwidth. Fixed sleep also inaccurate as precise timing; PTX nanosleep permits approximate duration [0,2*t], so exact deadlines require globaltimer recheck. https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf

No fullkernel inference, actual hardware bandwidth claim, calibration tuning, or chosen implementation. Potential alternative is coarse epoch/byte-grant accounting without per-packet CAS retry. Must enforce already-issued bytes, consumption deadlines, bounded burst, latency, sparse/idle behavior; no future-budget reservation or reference changes. Existing flat-combine batch model already tested elsewhere and is not claimed new here.
