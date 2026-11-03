# Fetch SM diagnosis, current v2, GPU3

Diagnostic work only. Production host `cab508dc`, math `34af`, FAR v2 `8507eab0` remain unchanged. See [report](../../../docs/mla_host_fetch_sm_diagnosis_20261103.md) and `analysis.json` for interpretation and limits.

- `trace.cuh`: whole-current-kernel phase/warp-wait/item timing in idle partial_o backing. 32 epochs ×128 CTAs ×40 uint64; only 108 CTAs active. Field mapping below.
- `capacity.cuh`: completed-remap attention replay, 84/108 active CTAs selected by host_item_entries840/1080 in the same binary. Every output row computed; after full kernel resets row cursors, exact unpadded length is restored from kv_indptr. No prepare/fetch/LRU in replay, and no official reference change. Grid barrier plus SMIDs validates108SM residency before24CTAs retire.
- `make_diagnostics.py` and `fix_replay_length.py` are one-time historical constructors, **not resume commands**. Initial capacity was corrected before any GPU replay. `pre_gpu_replay_length/` preserves first CPU build/source.
- `build.json` includes the old CPU build and final corrected build. Current per-variant hashes and actual library SHA are in `manifest.json`; resource dumps name the actual new typed kernels. Include trees/JIT artifacts are intentionally ignored.

GPU3 A100-PCIE NUMA1, same-card far HBM, not NVLink. BF16 h64 K2048 MTP8 R128 LRU4096, staged f24 i64 splitoff,256positions/32steps. Sources and trace bindings match fresh `rebench_100gb_v2_20261103`. CUDA Graph/events; uploads, reductions, references, checks outside attention events. Trace onepass each100GB0/20; replay threecoldpasses100GB0, alternating84/108 order perstep/pass.

Reproduce only when a new diagnostic is justified; do not overwrite accepted files. Rebuild private include trees if absent by copying current repo include after verifying production/math/FAR manifest hashes, then replace only host header with frozen trace/capacity.cuh. Root Python `/mnt/data/mhy-vllm-venv/bin/python`, CUDA13.1 arch8.0, PYTHONPATH root, TMPDIR root/temp. `build.py` is CPU-only and builds current frozen headers. `run.py` preflights stable10s physicalGPU3 idle, child holds shared gpu3_correctness.lock and bindsNUMA1. It skips complete accepted outputs. Each setting gets a separate process because TieredKV retains VMM mappings until explicitclose/process exit. No old runner was restarted.

`first_driver/` preserves the first0µs successful trace followed by allocationOOM before20µs; that completed raw remains accepted. Only missing20µs and capacity were then executed. Logs are preserved. All actions terminal, no waiter. `analyze.py` validates final hashes/resources/provenance, all outputs/state and cache payloads, and keeps overlapping time budgets distinct.

Trace fields (timestamps nanoseconds, counts/unit sums as indicated):

| Index | Meaning |
|---|---|
|0|CTA entry|
|1/2|before/after prepare gridbarrier|
|3/4|resolve done / after second gridbarrier|
|5/6|fetch branch start / return (all CTAs)|
|7/8/9|items begin / end / before exit-reset|
|10/11/12|SMID / epoch / fetcher flag|
|13/14|attention/LRU item counts|
|15/16|LRU/attention perCTA nanosecond sums|
|17/18/19|warp0 wait-call total ns / calls / calls with actual blocked iteration|
|20/21/22|first attention start / last attention start / last attention end|
|23/24/25|first ready / last ready / fetchloop end before deferred cache|
|26/27|physical cp.async wait / FAR wait perCTA ns sums|
|28/29|deferred chunks / fetch chunks|
|34/35/36|temporary begin timestamps|

Globaltimer has roughly1us resolution and tracing changes codegen (traceREG252, productionREG254; bothstack0). Actual traceevents +2.81%/+1.41% versus formal3pass averages, so not a production benchmark. SumCTA ns /108 is a service budget, not independently removable walltime. Wait calls include already-ready checks, instrumentation, and do not capture all barrier waits. Capacity3.048vs2.442ms confirms material scaling, but does not strictly assign0.606ms to the formal3.327ms result. No physicalNVLink/H800/newproduction gain or110% achievement.
