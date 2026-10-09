# CTA allocation experiment

## Result

Redistributing CTA work did not improve 25% or 30% Host hits in the tested configurations. The default remains 16 local and 32 Host compute partitions. Automatic allocation switching is not enabled.

The 44-local/32-Host configuration uses all 108 CTAs: 76 compute partitions and 32 merge CTAs. Across 128K-1M contexts, its 25% latency increases by 3.7%-4.9%, and its 30% latency increases by 4.0%-5.0%, against the paired frozen segmented reference.

## Complete paired comparison

Random placement, BF16, one decode query, 64 heads, top-k 2048. Forty rounds, eight index sets, 100 serialized cycles per graph. Times are median GPU cycles in us. Reference and candidate share identical inputs and physical KV allocations within each run; method order is randomized.

| Context | Reference 25% | 44/32 25% | Change | Reference 30% | 44/32 30% | Change |
|---|---:|---:|---:|---:|---:|---:|
| 128K | 36.96 | 38.33 | +3.7% | 41.46 | 43.17 | +4.1% |
| 256K | 34.94 | 36.63 | +4.9% | 40.24 | 42.05 | +4.5% |
| 512K | 36.97 | 38.34 | +3.7% | 42.60 | 44.29 | +4.0% |
| 1M | 34.88 | 36.22 | +3.8% | 40.05 | 42.05 | +5.0% |

## Additional allocation candidates

Each entry uses its own paired reference. Positive changes mean slower execution. CTAs are listed as local/Host compute partitions.

| Allocation | Merge reads | Rounds | 25% change | 30% change | Artifact |
|---|---|---:|---:|---:|---|
| 24/32 | Bulk | 12 | +3.2% | +5.0% | [fixed_local24_host32_random](fixed_local24_host32_random.json) |
| 32/32 | Bulk | 40 | +0.2% | +2.2% | [full_local32_host32_random](full_local32_host32_random.json) |
| 44/32 | Bulk | 40 | +3.8% | +5.0% | [full_local44_host32_random](full_local44_host32_random.json) |
| 16/48 | Bulk | 12 | +8.1% | +4.9% | [fixed_local16_host48_random](fixed_local16_host48_random.json) |
| 44/32 | 8 partitions per batch | 40 | +5.7% | +4.8% | [full_stream_local44_host32_random](full_stream_local44_host32_random.json) |

The 32/32 allocation improves 0%-15% in its 1M paired run:

- 0%: 27.69 to 22.39 us (-19.1%).
- 10%: 28.15 to 23.13 us (-17.8%).
- 15%: 27.88 to 26.67 us (-4.4%).
- 20%: 30.68 to 30.79 us (+0.4%).

These lower-hit-rate gains do not establish a benefit at 25%-30%.

## Critical path evidence

Globaltimer traces come from a separately instrumented 16/32 kernel, measured from the earliest CTA entry. Times below are in us. They are diagnostic single-call medians, have coarse timer granularity, and are excluded from the latency comparison. Individual medians need not add up.

| Host hits | Local compute complete | Host compute complete | Merge complete | Tail after compute |
|---|---:|---:|---:|---:|
| 25% | 17.4 | 29.7 | 32.8 | 4.1 |
| 30% | 17.4 | 34.8 | 38.9 | 4.1 |

Local compute already completes well before the Host side. Accelerating local compute alone does not shorten the observed critical path. Increasing partitions also writes more partial outputs and increases merging work. The traces do not isolate PCIe saturation or pure computation, so they cannot prove that link bandwidth is the only limit.

Some larger bulk-merge candidates introduce register spills. Batching merge reads removes spills in the final 44/32 streamed candidate (238 registers, zero stack), but that candidate still regresses 25% and 30%. Further work should shorten Host-dependent reading, computation, and merging rather than target SM occupancy alone.

## Verification and provenance

- GPU 2: A100-SXM4-40GB, Host NUMA node 1. GPU 1 had substantial allocations from other processes. All reference/candidate comparisons use the same GPU. Absolute timings vary across separate allocations; use paired changes when comparing trials.
- Frozen reference: source commit `da4ede5f507b50af6078ac5f0887bc5d773c031f`, module SHA256 `94015399d323f4755ec133b5a594dd9e5399e6dd07777b2c0017a691036133b8`. Local binary copy: `baseline_segmented.so`.
- The generalized 16/32 control matches the frozen binary within 0.22% across six hit rates. Both use 242 registers and zero stack.
- All 432 paired FP32 output checks passed; maximum absolute error 0.003636.
- Extended BF16/FP16 outputs, base-2/base-e LSE, random/head/tail placements, graph replay, Host mutations, tile boundaries, and the maximum 84-partition scratch layout passed 420 checks. This includes the four-head merge path.
- Racecheck: zero hazards/errors/warnings. Synccheck: zero errors.
- Original Memcpy/Direct/Segmented reports, figures, and raw datasets remain the baseline measurements. This experiment does not replace them.
- Initial scouting results and register reports are retained. Applying `before_loop_fix.patch` with `git apply --unidiff-zero` to the frozen reference source recreates the initial kernel used by the `stream_*` scouting runs. Its dynamic head loop introduced extra spills. The final compile-time pass loop restores the 16/32 control and is used by the primary measurements.

[Validation manifest](verification.json) · [Phase samples](phases.json) · [Original comparison](../host_attention_comparison/report.md)

## Reproduce the complete comparison

```bash
PYTHONPATH=. FLASHINFER_WORKSPACE_BASE="$PWD/temp/jit_host_balance" \
  FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR="$PWD/temp" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=4 \
  /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_comparison.py \
    --device 2 --cpu-affinity 32-63,96-127 \
    --host-kernel private --private-chunks 32 --private-pipeline \
    --local-chunks 44 --contexts 131072 262144 524288 1048576 \
    --fractions 0 0.1 0.15 0.2 0.25 0.3 --rounds 40 --index-sets 8 \
    --baseline-module temp/host_attention_balance/baseline_segmented.so \
    --output temp/host_attention_balance/new_full_local44_host32_random.json
```
