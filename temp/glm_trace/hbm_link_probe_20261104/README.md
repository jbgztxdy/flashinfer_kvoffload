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
