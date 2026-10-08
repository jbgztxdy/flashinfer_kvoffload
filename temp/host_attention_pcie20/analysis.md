# PCIe budget for 25% and 30% Host hits

## Scope and target

GPU 1 is an A100-SXM4-40GB on a PCIe 4.0 x16 link. The experiment uses BF16 MLA, 64 query heads, one decode token, and top-k 2048. Each selected KV row has 512 latent features plus 64 RoPE features: 1152 bytes. Index selection and top-k scoring are excluded. Host memory is pinned and NUMA-local to GPU 1.

The native HBM reference at 1M context is 27.463679 us. A 20% latency increase permits 32.956415 us. A 20% throughput reduction permits 34.329599 us. We use the stricter latency target in the tables below.

## Link and transfer budget

PCIe 4.0 is 16 GT/s per lane with 128b/130b encoding. One-way x16 capacity before packet overhead is 31.507692 GB/s. The frequently quoted 64 GB/s aggregates both directions and does not double the Host-to-GPU read budget.

Measured large, contiguous GPU reads from mapped Host memory sustain 27.032958 GB/s on this machine. This is a stream proxy, not a guarantee for sparse reads.

| Host hits | Rows | Payload | Link-only minimum (us) | Stream proxy (us) | Sparse read-and-store (us) | Remaining after stream proxy (us) |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 512 | 576 KiB | 18.72 | 21.82 | 25.41 | 11.14 |
| 30% | 614 | 690.75 KiB | 22.45 | 26.17 | 29.76 | 6.79 |

The sparse reference reads the same selected rows from Host and writes them to an HBM buffer. It includes startup, index processing, and the store, and excludes attention. It is an implementation reference, not a physical lower bound. Its 1M, 64-CTA medians leave 7.55 us and 3.19 us within the latency budget, respectively. Changing context from 128K to 1M has little effect on this reference.

Bandwidth alone does not rule out either target. Full overlap would make the maximum of the local work and transfer time dominant. However, the native HBM time includes HBM loads, reductions, synchronization, and kernel scheduling. It is not a measurement of pure computation, so using it as the available compute window is optimistic. The attention still has to finish Host-dependent QK, softmax, PV, and output merging after the final required data arrives.

## Measured root cause and changes

The stable fused baseline reads selected Host rows into an HBM scratch buffer using producer warps. Host computing CTAs wait for rows, copy scratch data into shared memory, then perform attention. Final merging waits for the Host results. Those dependencies leave work after the PCIe reads; available link bandwidth by itself does not remove this tail.

The first private-shared prototype reads Host rows directly into the computing CTA's shared memory with cp.async. It removes KV HBM staging and cross-CTA KV readiness flags. Using 60 Host splits only modestly improves 25% and 30%; the larger number of partial outputs increases merge overhead and regresses lower hit rates. It is retained as an experiment and was not selected.

The selected version uses 32 Host splits and reads KV in three groups: the first 256 latent dimensions, the second 256 latent dimensions, and RoPE. QK accumulation starts when each required group is available, while later groups are in flight. It preserves the complete softmax and PV calculation and all selected rows.

## Evidence

- `bandwidth_budget.json`: exact budget calculations and reference values.
- `sparse_fetch.json`: 24 measured sparse read cases with output verification.
- `../mapped_host_read_bandwidth.csv`: original mapped Host stream measurements.
- `private_v1.json`: paired private-shared prototype and frozen baseline measurements.
- `baseline_v44.json`, `baseline_v44.so`: preserved stable baseline identity and binary.

Sources:
- https://pcisig.com/sites/default/files/2026-01/30%20Years%20of%20PCI-SIG%20Innovation%20Webinar_Updated%203.23.2023.pdf
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory

## Optimization outcome

The selected version is now the default `host_page_start` path in `BatchMLAPagedAttentionWrapper`. The staged producer baseline remains available through `FLASHINFER_MLA_HOST_STAGED_BASELINE`, as well as the frozen V44 module. Its existing performance results are preserved.

In the final random-layout comparison, 25% decreases from 38.80-38.87 us to 34.89-35.07 us, and 30% from 43.13-43.63 us to 40.06-40.67 us. The HBM reference remains 27.37-27.40 us. The corresponding throughput loss is about 22% and 32%, so neither the latency +20% nor throughput -20% target has been met. The smaller partial QK window does not hide the full Host-dependent softmax, PV, output stores, and final merge. Kernel startup and index preparation also consume budget.

The initial complete run encountered external activity on GPU 1: memory rose above the benchmark's allocation and all-local times increased from about 27 us to 32-34 us. The affected random 1M and head 128K/256K contexts were rerun, with HBM references returning to about 27.4 us. Initial raw files, reruns, and replacement provenance are all retained. No samples were trimmed from the individual 40-round runs.

## Exploratory variants

These trials used 12 rounds on 1M context, with identical inputs between methods within each run. Different runs experienced different physical memory and shared-machine conditions; use the paired V44 control when assessing each trial. Complete final measurements use 40 rounds and appear in report.md.

| Trial | Change | V44 25% | Trial 25% | V44 30% | Trial 30% | Artifact |
|---|---|---:|---:|---:|---:|---|
| V1 | 60 private Host splits | 39.14 | 38.16 | 43.77 | 42.08 | private_v1.json |
| V2 | 28 splits with segmented QK | 38.84 | 37.40 | 43.32 | 41.88 | private_v2.json |
| V3 | 32 splits with segmented QK (selected) | 38.80 | 34.88 | 43.12 | 40.06 | private_v3.json |
| V4 | 40 splits with segmented QK | 40.65 | 38.97 | 45.72 | 43.65 | private_v4.json |
| V5 | Registered Host loads | 40.71 | 37.15 | 45.54 | 42.11 | private_v5.json |
| V6 | Earlier Host issue and merge preloads | 38.82 | 35.06 | 43.30 | 40.38 | private_v6.json |
| V7 | Duplicated QK without group exchange | 38.81 | 35.38 | 43.24 | 41.02 | private_v7.json |
| V8 | Warp-coalesced Host row loading | 38.81 | 35.40 | 44.59 | 40.57 | private_v8.json |
