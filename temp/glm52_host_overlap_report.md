# Mixed HBM and pinned Host DRAM MLA overlap experiment

## Scope and measured outcome

A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; latent rank 512; RoPE 64; page size one; top-k 2048. A single attention layer is tested; model projections, the sparse indexer, and other layers are excluded. Total cache contexts: 131072, 262144, 524288, 1048576 tokens. The first physical half resides in GPU HBM and the second in pinned NUMA-0 Host DRAM, mapped into one CUDA VMM address range. Host selections occupy random, head, or tail positions among the selected 2048 keys; physical token addresses remain sampled across each tier. Requested hit fractions are rounded to the nearest selected row; the 20% case contains 410 of 2048 rows (20.0195%).

The acceptance assumption is at most 5% latency increase relative to the same operator with 0% Host hits. The table aggregates the median of 12 independently measured context/layout medians; the range and per-case results below retain variation.

**The 20% target is not met.** At 10% and 15%, aggregate medians are close to the new operator's local baseline; the worst 15% context/layout case exceeds 5%. The new operator's 0% latency is also approximately 2.6% above the native local baseline. At high Host fractions, the fused design regresses relative to native direct attention; native execution remains available as the baseline.

## Latency and transfer model

| Host hits | Host keys | Native direct (us) | Fused overlap (us) | Fused / its 0% | Transfer at 27 GB/s (us) |
|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 27.41 | 28.13 | +0.0% | 0.00 |
| 10% | 205 | 31.80 | 28.84 | +2.9% | 8.75 |
| 15% | 307 | 34.08 | 28.81 | +2.7% | 13.10 |
| 20% | 410 | 38.20 | 32.81 | +16.6% | 17.49 |
| 30% | 614 | 46.26 | 45.28 | +60.9% | 26.20 |
| 40% | 819 | 54.77 | 55.17 | +96.2% | 34.94 |
| 60% | 1229 | 72.24 | 75.89 | +169.8% | 52.44 |
| 80% | 1638 | 89.72 | 97.26 | +245.5% | 69.89 |
| 100% | 2048 | 107.21 | 117.97 | +319.3% | 87.38 |

The 27 GB/s number is measured mapped-memory streaming bandwidth. The PCIe 4.0 x16 coding-adjusted one-way ceiling is 31.51 GB/s. The transfer columns omit launch, sparse addressing, synchronization, and attention arithmetic. The isolated sparse-fetch microbenchmark takes approximately 21 us at 20% (410 rows, 472320 bytes), so the 17.49 us streaming estimate is optimistic for this workload. Neither figure proves complete overlap.

## Implementation and remaining tail

The opt-in `host_page_start` path adds one fused cooperative CUDA attention kernel. It partitions selected keys into 16 local compute tasks and 28 Host compute tasks. The other 64 CTAs use one warp group to issue independent packed PCIe reads and another to merge local attention states. Epoch-tagged ready flags publish staged rows and partial attention results with release/acquire ordering. Host attention uses tensor cores and a one-tile path for small chunks. CPU values retain their original BF16/FP16 representation.

The kernel reads mapped Host memory directly, then writes those rows into internal HBM scratch before tensor-core attention consumes them. This buffering happens inside attention. There is no external gather kernel or CUDA memcpy in the timed path. Internal allocation is 5.25 MiB of HBM plus 32 KiB synchronization state, allocated and initialized during warmup.

Final phase instrumentation (`glm52_overlap_host_final_phase_summary.json`) shows local computation ending around 21.5 us, Host rows becoming ready around 22.5 us, and Host partial writes ending at 25.6-27.6 us at 20%. Thus CPU reads overlap local computation, while Host attention and the final reduction still extend the critical path. The timer has approximately 1 us granularity, and these are separate debug runs, not the final latency measurements. A write-combined Host allocation was also measured; it did not materially improve sparse transfer time (`sparse_host_allocators.json`).

The trial history keeps rejected designs, including scalar final PV, duplicated QK, smaller/larger tiles, staged probability sharing, and parallel partial reductions. The final implementation restores the best balanced packed-load design and adds explicit warp synchronization around shared output conversion. Racecheck originally identified warp-local shared-memory write/read hazards in the reused native output helper; the Host path now synchronizes before reading and before reusing that tile. Native calls retain their original default path.

## Timing and correctness

Each case uses 40 rounds, eight distinct index sets, 100 attention calls per hot CUDA Graph, 20 eager calls, and a separate one-call measurement after 64 MiB L2 eviction. Host fractions are shuffled in each round. Top-k selection, plan, index updates, cache initialization, graph construction, and independent FP32 references are excluded. Host reads, HBM staging, attention computation, and output merging are included. Eager GPU events also contain CPU launch gaps; cold single-call timing includes graph launch gaps.

`glm52_overlap_correctness.json` records 228 independent FP32 comparisons covering BF16/FP16, all fractions, random/head/tail layouts, two seeds, base-2/base-e LSE, graph reuse, and CPU-side Host mutations. `glm52_overlap_api_correctness.json` records native page-size-16, causal, multi-request, multi-query, 128-head regressions and rejected invalid Host inputs. Sanitizer logs are retained separately. The nine historical baseline hashes are preserved in `glm52_host_historical_baselines_manifest.json`.

`glm52_overlap_host_direct_verification.json` combines a CUDA graph node trace with a system-memory traffic counter. The timed graph contains exactly 100 attention kernels, zero GPU memcpy activities, and zero memset activities. At 20%, the final kernel reads 14760 system-memory fill sectors of 32 bytes each, exactly 472320 bytes for 410 selected Host rows. The profiled durations are excluded from the performance table. Resource inspection records 238 registers and zero stack/local memory for the BF16 kernel.

GPU 1 was selected for serial measurements. Other GPUs ran unrelated jobs, and GPU 1 contained two idle EngineCore processes. Before/after GPU snapshots, source hashes, and the Git commit are stored in each result. Hardware was shared; these results do not establish single-tenant production behavior or H800 performance.

## Per-context and layout results

### random

| Context | Host | Hot (us) | Increase | Cold single (us) | Eager (us) |
|---:|---:|---:|---:|---:|---:|
| 131072 | 0.0% | 27.80 | 0.0% | 37.89 | 31.13 |
| 131072 | 10.0% | 28.85 | 3.8% | 39.94 | 34.43 |
| 131072 | 15.0% | 28.81 | 3.6% | 41.98 | 34.43 |
| 131072 | 20.0% | 32.77 | 17.9% | 46.08 | 38.40 |
| 131072 | 30.0% | 45.28 | 62.9% | 55.30 | 51.00 |
| 131072 | 40.0% | 55.07 | 98.1% | 64.51 | 60.54 |
| 131072 | 60.0% | 75.88 | 172.9% | 86.02 | 81.59 |
| 131072 | 80.0% | 97.31 | 250.0% | 107.01 | 102.99 |
| 131072 | 100.0% | 117.85 | 323.9% | 128.00 | 123.57 |
| 262144 | 0.0% | 28.13 | 0.0% | 37.89 | 31.08 |
| 262144 | 10.0% | 29.22 | 3.9% | 39.94 | 34.28 |
| 262144 | 15.0% | 28.99 | 3.1% | 40.96 | 34.25 |
| 262144 | 20.0% | 32.75 | 16.4% | 46.08 | 38.20 |
| 262144 | 30.0% | 45.28 | 60.9% | 55.30 | 50.74 |
| 262144 | 40.0% | 55.20 | 96.2% | 64.51 | 60.42 |
| 262144 | 60.0% | 75.90 | 169.8% | 86.02 | 81.36 |
| 262144 | 80.0% | 97.30 | 245.8% | 106.50 | 102.76 |
| 262144 | 100.0% | 117.96 | 319.3% | 126.98 | 123.24 |
| 524288 | 0.0% | 28.11 | 0.0% | 36.86 | 30.41 |
| 524288 | 10.0% | 29.18 | 3.8% | 38.40 | 33.64 |
| 524288 | 15.0% | 28.97 | 3.0% | 39.94 | 33.59 |
| 524288 | 20.0% | 32.71 | 16.3% | 45.06 | 37.40 |
| 524288 | 30.0% | 45.24 | 60.9% | 54.27 | 50.07 |
| 524288 | 40.0% | 55.15 | 96.2% | 63.49 | 59.65 |
| 524288 | 60.0% | 75.88 | 169.9% | 83.97 | 80.69 |
| 524288 | 80.0% | 97.26 | 245.9% | 105.47 | 102.09 |
| 524288 | 100.0% | 117.92 | 319.5% | 125.95 | 122.62 |
| 1048576 | 0.0% | 28.19 | 0.0% | 37.89 | 31.13 |
| 1048576 | 10.0% | 29.26 | 3.8% | 39.94 | 34.41 |
| 1048576 | 15.0% | 29.04 | 3.0% | 41.98 | 34.46 |
| 1048576 | 20.0% | 32.81 | 16.4% | 46.08 | 38.40 |
| 1048576 | 30.0% | 45.34 | 60.8% | 55.30 | 50.94 |
| 1048576 | 40.0% | 55.23 | 95.9% | 64.51 | 60.62 |
| 1048576 | 60.0% | 75.97 | 169.5% | 86.02 | 81.56 |
| 1048576 | 80.0% | 97.36 | 245.4% | 106.50 | 102.99 |
| 1048576 | 100.0% | 118.16 | 319.1% | 128.00 | 123.65 |

### head

| Context | Host | Hot (us) | Increase | Cold single (us) | Eager (us) |
|---:|---:|---:|---:|---:|---:|
| 131072 | 0.0% | 27.91 | 0.0% | 40.96 | 31.13 |
| 131072 | 10.0% | 28.82 | 3.2% | 44.03 | 34.30 |
| 131072 | 15.0% | 29.80 | 6.8% | 46.08 | 34.48 |
| 131072 | 20.0% | 33.34 | 19.5% | 50.18 | 39.01 |
| 131072 | 30.0% | 46.04 | 65.0% | 58.88 | 51.51 |
| 131072 | 40.0% | 55.46 | 98.7% | 67.58 | 61.03 |
| 131072 | 60.0% | 76.71 | 174.9% | 90.11 | 81.95 |
| 131072 | 80.0% | 98.40 | 252.6% | 110.59 | 102.96 |
| 131072 | 100.0% | 119.62 | 328.6% | 135.68 | 124.29 |
| 262144 | 0.0% | 28.21 | 0.0% | 38.91 | 31.08 |
| 262144 | 10.0% | 28.87 | 2.3% | 41.98 | 34.30 |
| 262144 | 15.0% | 28.85 | 2.3% | 44.03 | 34.53 |
| 262144 | 20.0% | 33.01 | 17.0% | 48.64 | 38.63 |
| 262144 | 30.0% | 46.03 | 63.2% | 57.34 | 51.51 |
| 262144 | 40.0% | 55.37 | 96.3% | 66.56 | 60.95 |
| 262144 | 60.0% | 76.63 | 171.7% | 88.06 | 82.10 |
| 262144 | 80.0% | 97.44 | 245.5% | 111.62 | 103.27 |
| 262144 | 100.0% | 118.45 | 319.9% | 130.05 | 123.83 |
| 524288 | 0.0% | 28.13 | 0.0% | 36.86 | 30.46 |
| 524288 | 10.0% | 28.79 | 2.3% | 37.89 | 33.59 |
| 524288 | 15.0% | 28.78 | 2.3% | 39.94 | 33.59 |
| 524288 | 20.0% | 32.80 | 16.6% | 45.06 | 37.63 |
| 524288 | 30.0% | 45.25 | 60.8% | 54.27 | 50.20 |
| 524288 | 40.0% | 55.19 | 96.2% | 63.49 | 59.55 |
| 524288 | 60.0% | 75.90 | 169.8% | 83.97 | 80.74 |
| 524288 | 80.0% | 97.16 | 245.3% | 105.47 | 102.02 |
| 524288 | 100.0% | 117.90 | 319.1% | 125.95 | 122.80 |
| 1048576 | 0.0% | 28.11 | 0.0% | 36.86 | 30.44 |
| 1048576 | 10.0% | 28.79 | 2.4% | 38.91 | 33.48 |
| 1048576 | 15.0% | 28.78 | 2.4% | 40.45 | 33.54 |
| 1048576 | 20.0% | 32.85 | 16.8% | 44.03 | 37.63 |
| 1048576 | 30.0% | 45.21 | 60.8% | 54.27 | 50.12 |
| 1048576 | 40.0% | 55.18 | 96.3% | 63.49 | 59.65 |
| 1048576 | 60.0% | 75.90 | 170.0% | 83.97 | 80.74 |
| 1048576 | 80.0% | 97.15 | 245.5% | 105.47 | 102.04 |
| 1048576 | 100.0% | 117.83 | 319.1% | 125.95 | 122.68 |

### tail

| Context | Host | Hot (us) | Increase | Cold single (us) | Eager (us) |
|---:|---:|---:|---:|---:|---:|
| 131072 | 0.0% | 27.80 | 0.0% | 37.89 | 31.33 |
| 131072 | 10.0% | 28.78 | 3.5% | 39.94 | 34.46 |
| 131072 | 15.0% | 28.75 | 3.4% | 41.47 | 34.46 |
| 131072 | 20.0% | 32.81 | 18.0% | 46.08 | 38.60 |
| 131072 | 30.0% | 45.28 | 62.9% | 55.81 | 51.05 |
| 131072 | 40.0% | 54.69 | 96.7% | 65.54 | 60.80 |
| 131072 | 60.0% | 75.88 | 172.9% | 86.02 | 81.89 |
| 131072 | 80.0% | 97.21 | 249.7% | 107.52 | 103.12 |
| 131072 | 100.0% | 117.87 | 324.0% | 128.00 | 123.75 |
| 262144 | 0.0% | 28.14 | 0.0% | 37.89 | 31.23 |
| 262144 | 10.0% | 28.84 | 2.5% | 40.45 | 34.36 |
| 262144 | 15.0% | 28.79 | 2.3% | 41.98 | 34.30 |
| 262144 | 20.0% | 32.80 | 16.6% | 47.10 | 38.40 |
| 262144 | 30.0% | 45.28 | 60.9% | 55.30 | 50.89 |
| 262144 | 40.0% | 54.67 | 94.3% | 65.54 | 60.57 |
| 262144 | 60.0% | 75.87 | 169.6% | 86.02 | 81.87 |
| 262144 | 80.0% | 97.22 | 245.5% | 107.52 | 103.01 |
| 262144 | 100.0% | 117.98 | 319.3% | 128.00 | 123.65 |
| 524288 | 0.0% | 28.15 | 0.0% | 37.89 | 31.13 |
| 524288 | 10.0% | 28.81 | 2.3% | 39.94 | 34.23 |
| 524288 | 15.0% | 28.79 | 2.3% | 41.47 | 34.23 |
| 524288 | 20.0% | 32.80 | 16.5% | 46.08 | 38.25 |
| 524288 | 30.0% | 45.28 | 60.8% | 55.30 | 50.87 |
| 524288 | 40.0% | 54.79 | 94.6% | 64.51 | 60.36 |
| 524288 | 60.0% | 75.88 | 169.5% | 86.02 | 81.59 |
| 524288 | 80.0% | 97.25 | 245.4% | 107.01 | 102.91 |
| 524288 | 100.0% | 118.01 | 319.1% | 128.00 | 123.39 |
| 1048576 | 0.0% | 28.20 | 0.0% | 37.89 | 31.28 |
| 1048576 | 10.0% | 28.85 | 2.3% | 39.94 | 34.51 |
| 1048576 | 15.0% | 28.81 | 2.1% | 41.98 | 34.46 |
| 1048576 | 20.0% | 32.81 | 16.4% | 46.59 | 38.55 |
| 1048576 | 30.0% | 45.31 | 60.7% | 55.30 | 51.02 |
| 1048576 | 40.0% | 54.73 | 94.1% | 65.54 | 60.77 |
| 1048576 | 60.0% | 75.89 | 169.1% | 86.02 | 81.82 |
| 1048576 | 80.0% | 97.25 | 244.8% | 107.52 | 103.17 |
| 1048576 | 100.0% | 118.02 | 318.5% | 128.00 | 123.75 |

## Reproduce

```bash
PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_overlap_benchmarks.py
python temp/summarize_glm52_host_overlap.py
```

Existing benchmark outputs are protected from overwrite. Use new output paths for a rerun.
