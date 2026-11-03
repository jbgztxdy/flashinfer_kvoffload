# Fetch QK assist feasibility probe — compiled, NOT GPU tested

See [design](../../../docs/mla_host_fetch_compute_design_20261103.md).

Current `probe.cu` compiled with CUDA13.1 SM80. Fixed missingcstdint in first_compile archive. Actual resources in resources.txt/build_fixed.log: purefetchREG30, assistREG54, stack/local0; both launch24CTA×256threads×147968dynamicshared. All production headers and v2 FAR remain unchanged.

`budget.py` runs CPU-only measured-budget accounting and operand/output coordinate coverage; budget.json binds frozen source/binary/production/FAR SHA. This is not GPU arithmetic validation.

`run.py` attempted stable10s GPU3-idle preflight under sharedgpu3lock; externalflashcheck remained active and120s preflight expired before CUDA/probe launch. first_preflight_run.py preserves executed driver; preflight_failure.json records failure/snapshot. Driver now saves each observation on future attempts. No result.json and no accepted performance/correctness result. No waiter remains. Do not infer a benefit from compiling54registers.

When an actual free window exists, from repo root:

```
/mnt/data/mhy-vllm-venv/bin/python -u temp/glm_trace/probe_fetch_qk_assist_20261103/run.py
```

Driver binds physicalGPU3 andNUMA1; never kill external tasks. Probe uses unchanged v2 reserve/wait hooks, double64row buffers, fixed100GB0/20µs. It times purefetch/fetch+oneQKblock-perchunk/standaloneQK, publishes actual scores, and checks allremote payload plus fourjobs×2048scores against CPU BF16-product FP32 accumulation. No actual worker consumes scores, no production work is removed, no LRU/concurrent84CTA. Timed standalone is a component cost, not a formal reference. A successful probe requires real integration to establish useful assistance or end-to-end speedup.

Frozen sources, CPU build failures and preflight failure must be retained. probe binary is ignored. Recompile only if absent after verifying source/FAR SHA:

```
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_80 -I include -Xptxas=-v temp/glm_trace/probe_fetch_qk_assist_20261103/probe.cu -o temp/glm_trace/probe_fetch_qk_assist_20261103/probe
```
