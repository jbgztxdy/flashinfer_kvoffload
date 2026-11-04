# Same-device HBM microbenchmark

GPU3/A100-PCIE, NUMA1. No MLA/FlashInfer JIT or far-link simulator. Two disjoint allocations on the same GPU; this does not traverse NVLink/PCIe. Standalone CUDA source probe.cu, Python build/preflight/output wrapper run.py. Compile requires CUDA13.1; target SM80.

Run from the repository root:

    /mnt/data/mhy-vllm-venv/bin/python temp/glm_trace/hbm_link_probe_20261104/run.py --wait-seconds 60

The wrapper holds the existing GPU3 shared lock and requires 10 consecutive seconds with no other compute process, less than512MiB and less than10% utilization. It never terminates another process. --compile-only builds without CUDA device allocation. GPU code runs via numactl -N1 -m1 with physical GPU3 mapped to logical device0. No automatic repeat overwrites previous output; each run has timestamped JSON/raw/log files.

Bandwidth: 262144 pages ×1152B =301989888B (288MiB) per copy, source/destination each larger than A100 L2. Three warmups, seven samples, ten copies per CUDA-event sample, all raw times preserved and median reported. Payload GB/s counts source bytes once, matching remote-payload bandwidth convention. Read-plus-write GB/s counts twice that; these two figures describe the same copy. Checks compare every destination uint4 outside timing.

- cudaMemcpyAsync D2D: library/runtime copy baseline.
- SM contiguous copy: a separate explicit vectorized copy kernel using all SMs.
- SM sparse unique copy: random unique source-page permutation, contiguous destination, 1152B pages, 64pages per packet, two73728B shared buffers, cp.async.cg, 256threads, 24CTA and oneCTA per SM. Actual copies and necessary writes included; no attention/cache-admission math. This resembles fetch shape but does not predict full MLA time.

Latency: one CUDA lane follows a shuffled dependent cycle of uint64 next indices at128B spacing, ld.global.cg bypassing L1. Small1MiB and large256MiB address working sets; 1024 warmup hops, then at least65536 hops and an entire ring for the large case, seven samples. A full large ring avoids accidentally caching only a short repeatedly traversed prefix. globaltimer measures total chain duration; CPU verifies the final index. Reported ns/hop includes dependent load/address/control cost; it is neither a pure DRAM pin latency nor CUDA launch time nor parallel KV-packet completion latency. The small case measures cache-resident dependency cost; the large case estimates random HBM load-use latency without profiler proof of every miss.

Resource metadata and actual source/binary SHA256 accompany each attempt. A not_run record means no GPU experiment was performed; it is not a bandwidth result. If an external process appears by the final check, results are not accepted as exclusive measurements. Initial compilation succeeded; the first execution attempt encountered busy GPU3, see timestamped result JSON. Production headers and all prior MLA results remain unchanged.

## Completed GPU3 measurement, 2026-11-04 10:03

User reported idle; 10s stable idle preflight and shared lock passed. Complete raw in raw_20261104_100309.jsonl, result_20261104_100309.json. Every copy destination vector and pointer-chain final index checked. Start/end resource query found no external GPU3 process. End utilization100% reflects the just-completed dependent-load benchmark; memory was0MiB and no foreign process, not an idle-performance claim. GPU A100-PCIE-40GB/108SM/40MiB L2, queried SM1410MHz/memory1215MHz at both endpoints. Seven samples per measurement, median:

| Copy method | Payload GB/s | Read+write GB/s | 288MiB copy us |
| --- | ---: | ---: | ---: |
| cudaMemcpyAsync D2D | 687.601 | 1375.202 | 439.194 |
| SM contiguous | 548.776 | 1097.551 | 550.298 |
| Sparse unique1152B/64pages/double buffer/24CTA | 207.480 | 414.960 | 1455.514 |
| Same sparse copy/108CTA | 596.625 | 1193.251 | 506.163 |

Dependent .cg load-use latency:1MiB address working set203.531ns/hop;256MiB random full-ring working set406.383ns/hop. The latter is approximately0.406us per serial dependent load including address/loop work, not a model's fixed latency for a concurrent packet. Small-ring result is L2-resident behavior, not HBM latency. This experiment has no profiler miss-rate measurement.

Thus same-device “unlimited” does not specify one fixed bandwidth: sparse24CTA payload throughput is around207GB/s in this probe while more parallel copy reaches597GB/s and D2D688GB/s. It is not physical NVLink. The sparse probe has one destination write; production additionally performs cache-admission/writeback, ready synchronization and concurrent attention, so these figures are isolated transport baselines rather than full-MLA achieved throughput or a replacement for its stepwise reference.
