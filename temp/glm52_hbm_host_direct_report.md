# Corrected MLA HBM / Host DRAM direct access benchmark

## Root cause of the previous comparison

- The historical `bench_glm52_sparse_decode_host_direct.py` allocated the entire KV cache with `cudaHostAllocMapped`. Its nominal 0% case was also entirely Host DRAM. Its fraction labels changed indices, not memory residency.
- That test used Python launches between GPU events; the original HBM benchmark used CUDA Graph replay. GPU 0 was shared with other work and recorded 100% utilization in historical snapshots. The old 111–241 us results must not be treated as HBM measurements or a controlled mixed-tier comparison.
- Historical scripts and raw results are preserved unchanged. Their hashes are in `glm52_host_historical_baselines_manifest.json`.

## Corrected implementation

- Original FlashInfer `BatchMLAPagedAttentionWrapper`, FA2, unchanged attention kernel.
- One CUDA VMM range: first half backed by GPU 1 HBM (`location_type=DEVICE`), second half by pinned CPU NUMA 0 DRAM (`location_type=HOST_NUMA`, `allocation_type=PINNED`). No managed-memory migration.
- `cuMemGetAllocationPropertiesFromHandle` verifies physical residency; each result records the returned properties. The CPU writes Host rows through a CPU view; attention directly reads both halves through the common device pointer.
- 0% selects HBM rows only; 100% selects Host rows only. Every case counts selected Host rows and verifies `round(fraction * 2048)`.
- No KV gather, staging kernel, or KV memcpy in timed attention graphs. Only 8 KiB of index metadata is updated before timing.
- This uses the [CUDA VMM Host NUMA allocation API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html) to combine both tiers under one cache base pointer.

## Hardware and timing

- Idle NVIDIA A100-SXM4-40GB GPU 1; Host NUMA node 0; SM 1410 MHz / HBM 1215 MHz in recorded snapshots.
- Batch 1, query length 1, 64 heads, BF16, compressed KV rank 512, RoPE 64, page size 1, top-k 2048.
- Total contexts: 131072, 262144, 524288, 1048576 tokens; half HBM and half pinned Host capacity.
- 40 randomized rounds, 8 index sets. Main result: 100 attention calls per CUDA Graph; GPU events outside replay; elapsed time divided by 100.
- Also records Python/eager GPU timeline time and one attention graph after 64 MiB L2 eviction. The single-call measurement includes graph submission gaps and is not the main steady-state metric.
- Excludes top-k computation, plan, initialization, index updates, and correctness reference. No overlap with transfers.

## Random layout: attention microseconds

| Host hits among top-k | 128K | 256K | 512K | 1M | 1M / HBM |
|---:|---:|---:|---:|---:|---:|
| 0% | 27.484 | 27.443 | 27.484 | 27.484 | 1.000x |
| 10% | 32.174 | 32.138 | 31.964 | 31.616 | 1.150x |
| 15% | 34.038 | 33.935 | 33.812 | 33.341 | 1.213x |
| 20% | 37.601 | 37.565 | 37.571 | 37.335 | 1.358x |
| 30% | 46.305 | 46.336 | 46.285 | 46.049 | 1.675x |
| 40% | 55.071 | 55.071 | 54.989 | 54.769 | 1.993x |
| 60% | 72.556 | 72.545 | 72.561 | 72.243 | 2.629x |
| 80% | 90.040 | 90.051 | 90.097 | 89.702 | 3.264x |
| 100% | 107.597 | 107.540 | 107.459 | 107.162 | 3.899x |

## 1M context: placement in the top-k list

Head and tail refer to the first or last positions in the selected top-k list. Physical token IDs within the Host half remain randomly sampled, as in the original NVLink benchmark.

| Host hits | Random | Head | Tail |
|---:|---:|---:|---:|
| 0% | 27.484 | 27.438 | 27.474 |
| 10% | 31.616 | 31.037 | 32.512 |
| 15% | 33.341 | 34.350 | 34.770 |
| 20% | 37.335 | 37.750 | 37.832 |
| 30% | 46.049 | 46.136 | 46.167 |
| 40% | 54.769 | 54.723 | 54.671 |
| 60% | 72.243 | 72.192 | 72.182 |
| 80% | 89.702 | 89.723 | 89.754 |
| 100% | 107.162 | 107.192 | 107.203 |

## Validation

- All 108 context/placement/fraction cases passed an FP32 attention reference (rtol=0.01, atol=0.005). Maximum absolute output error: 0.003397.
- Reference rows come independently from HBM and the CPU Host view rather than an attention staging buffer.
- Nsight Systems captured exactly 100 MLA attention kernels in the timed 100%-Host graph. No CUDA memcpy activity table or gather/staging kernel was present. See `glm52_hbm_host_direct_trace_validation.json` and the `.nsys-rep` artifact.
- Profiled timing is for path validation and is excluded from the performance tables.

## Preserved historical baselines

| Artifact | Measurement / limitation |
|---|---|
| `glm52_sparse_decode_host_memcpy_full.json` | Explicit Host selection/transfer plus HBM staging attention. The copy region includes CPU indexing and temporary buffer creation; it is not pure pinned DMA bandwidth. |
| `host_mapped_smoke.json` | Mapped Host source read by a separate GPU gather/copy stage, then HBM attention. Not direct attention. One round with contaminated hot timings; retain as historical evidence. |
| `glm52_sparse_decode_host_direct.json` | All-Host direct attention with Python launches. Nominal fractions do not represent HBM/Host residency. |
| `glm52_sparse_decode_p2p*.json` | Original NVLink direct remote-HBM baselines; retained. |

## Reproduce

Run serially; do not overlap GPU benchmarks:

```bash
for layout in random head tail; do
  PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_sparse_decode_hbm_host_direct.py --device 1 --layout "$layout" --output "temp/glm52_hbm_host_direct_${layout}_rerun.json"
done
```

The script rejects an existing output path to protect prior results.

GPU events around Python launches include GPU idle gaps between submissions. Corrected 1M random all-HBM: 27.484 us Graph versus 29.389 us eager; all-Host: 107.162 us Graph versus 111.053 us eager. Timing mode alone does not explain the old 0% label: memory placement was the primary error.

## Host direct-read bandwidth and estimated transfer time

The current CPU/GPU link is PCIe Gen4 x16. Its single-direction coding-adjusted ceiling is 31.5077 GB/s, before packet overhead. The advertised 64 GB/s is the rounded aggregate of both directions and is not a one-way read rate. Calculation: 16 GT/s * 16 lanes * (128/130) / 8. See the [PCI-SIG speed table](https://pcisig.com/sites/default/files/2026-01/30%20Years%20of%20PCI-SIG%20Innovation%20Webinar_Updated%203.23.2023.pdf).

The new standalone `bench_mapped_host_read.cu` kernel directly reads CUDA mapped pinned Host allocations. It does not time memcpy. On the same GPU 1, large contiguous 64/256 MiB reads achieve 27.00–27.03 GB/s, with all checksums verified. Data exceeds A100's 40 MiB L2; initialization is CPU-bound to NUMA-0 CPUs. See `mapped_host_read_bandwidth.csv`.

For top-k 2048, BF16 KV rank 512 + RoPE 64, total KV is 2,359,296 bytes = 2.25 MiB. Host bytes are `round(2048 * fraction) * 1152`. All time columns below are microseconds. The measured attention values are the prior unprofiled 1M random-layout results, not new attention runs.

| Host hits | Host KV (KiB) | Pull at link ceiling (31.51 GB/s) | Pull at measured streaming rate (27.03 GB/s) | Ideal full-overlap proxy* | Serial proxy* | Measured attention | Relative to HBM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.00 | 0.00 | 0.00 | 27.48 | 27.48 | 27.48 | 1.00x |
| 10% | 230.62 | 7.50 | 8.74 | 27.48 | 36.22 | 31.62 | 1.15x |
| 15% | 345.38 | 11.22 | 13.08 | 27.48 | 40.57 | 33.34 | 1.21x |
| 20% | 461.25 | 14.99 | 17.47 | 27.48 | 44.96 | 37.34 | 1.36x |
| 30% | 690.75 | 22.45 | 26.17 | 27.48 | 53.65 | 46.05 | 1.68x |
| 40% | 921.38 | 29.94 | 34.90 | 34.90 | 62.39 | 54.77 | 1.99x |
| 60% | 1382.62 | 44.94 | 52.37 | 52.37 | 79.86 | 72.24 | 2.63x |
| 80% | 1842.75 | 59.89 | 69.80 | 69.80 | 97.29 | 89.70 | 3.26x |
| 100% | 2304.00 | 74.88 | 87.27 | 87.27 | 114.76 | 107.16 | 3.90x |

*The full-overlap proxy is `max(27.484 us HBM attention, Host pull at streaming bandwidth)`. The serial proxy is their sum. The HBM baseline includes memory access and is not pure compute time; these are simplified estimates, not a measured compute/communication split. Contiguous streaming bandwidth does not guarantee sparse MLA transaction performance.

### Can direct loads overlap compute?

Yes. The captured MLA specialization uses two stages (`KernelTraits<false, 2, ...>`). At `include/flashinfer/attention/mla.cuh:877`, `load_kv` queues asynchronous global-to-shared loads, commits both stages, and the loop waits only for the stage it consumes. While QK, softmax and PV compute on one tile, the next tile's Host read may remain in flight. `load_128b_async` issues `cp.async.cg.shared.global` through `include/flashinfer/cp_async.cuh`. This is internal shared-memory buffering within the attention kernel, not an extra gather kernel or a Host-to-HBM memcpy. NVIDIA documents [global-to-shared asynchronous copy/compute overlap](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory).

There is no guarantee of full overlap: a tile's computation must wait for its own inputs; two stages limit prefetch distance; pipeline startup/drain, waits, sparse request concurrency, and synchronization can leave exposed latency. At 100% Host, the link-only bound is 74.88 us; the measured-streaming estimate is 87.27 us; the full-overlap proxy is 87.27 us; the serial proxy is 114.76 us; measured attention is 107.16 us. This is consistent with partial overlap and other overheads, and does not establish an exact overlap percentage. Small-fraction streaming estimates below the HBM baseline also do not imply all transfer latency can be hidden.

Derived numbers and assumptions are recorded in `glm52_hbm_host_direct_bandwidth_model.json`. Historical attention measurements and baseline files remain unchanged.
