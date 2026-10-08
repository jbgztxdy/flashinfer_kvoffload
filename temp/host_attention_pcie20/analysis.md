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

The first private-shared prototype reads Host rows directly into the computing CTA's shared memory with cp.async. It removes KV HBM staging and cross-CTA KV readiness flags. Using 60 Host splits only modestly improves 25% and 30%; the larger number of partial outputs increases merge overhead and regresses lower hit rates. It is retained as an experiment, with the stable path still the default.

The next prototype reduces Host splits and reads KV in three groups: the first 256 latent dimensions, the second 256 latent dimensions, and RoPE. QK accumulation starts when each required group is available, while later groups are in flight. It preserves the complete softmax and PV calculation and all selected rows.

## Evidence

- `bandwidth_budget.json`: exact budget calculations and reference values.
- `sparse_fetch.json`: 24 measured sparse read cases with output verification.
- `../mapped_host_read_bandwidth.csv`: original mapped Host stream measurements.
- `private_v1.json`: paired private-shared prototype and frozen baseline measurements.
- `baseline_v44.json`, `baseline_v44.so`: preserved stable baseline identity and binary.

Sources:
- https://pcisig.com/sites/default/files/2026-01/30%20Years%20of%20PCI-SIG%20Innovation%20Webinar_Updated%203.23.2023.pdf
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-copy-from-global-memory-to-shared-memory
