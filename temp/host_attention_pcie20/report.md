# Host attention: PCIe budget and segmented reads

A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; 512 latent features; 64 RoPE features; page size 1; top-k 2048. The cache has equal HBM and pinned NUMA-0 Host capacity. Context labels are total KV lengths; only the selected rows are read.

## PCIe budget

One-way PCIe 4.0 x16 coding ceiling: 31.51 GB/s. Measured large mapped-Host reads: 27.03 GB/s. The historical HBM reference is 27.46 us; latency +20% permits 32.96 us, whereas throughput -20% permits 34.33 us.

| Host | Payload (KiB) | Link-only minimum (us) | Stream proxy (us) | Sparse read/store (us) | Remaining after stream proxy (us) |
|---:|---:|---:|---:|---:|---:|
| 25% | 576.00 | 18.72 | 21.82 | 25.41 | 11.14 |
| 30% | 690.75 | 22.45 | 26.17 | 29.76 | 6.79 |

Bandwidth alone permits the target in an ideal schedule. The stream values omit sparse-read overhead and startup; the read/store benchmark includes its HBM destination write and is not a physical lower bound. Native HBM time includes memory and synchronization, so it is not all available computation to overlap. See [analysis](analysis.md) for the assumptions and sources.

## Methods and measurement

- Memcpy: CPU packing into a pinned buffer, followed by actual serial H2D DMA and native attention. GPU time includes DMA, handoff, and attention; CPU packing is reported separately in the CSV.
- Direct: the original attention kernel directly reads mapped Host memory.
- Segmented: computing CTAs read Host KV directly into shared memory in three groups and accumulate QK as each group arrives. It uses 16 local and 32 Host compute partitions. Internal HBM scratch holds attention partial results. No KV staging kernel or memcpy is used.
- Staged: the preserved V44 fused producer path, with 16 local and 28 Host compute partitions; its frozen module is loaded for the old/new comparison.
- All four methods use identical queries, KV contents, and selected indices; Staged and Segmented share scratch and epoch buffers. Forty rounds, eight index sets, 100 serial cycles per CUDA graph, shuffled method/fraction order. Values are medians in microseconds. Top-k selection, index updates, planning, initialization, and correctness references are excluded.
- Random/head/tail denotes Host positions within the selected 2048-row list. Physical addresses are sampled across each tier.
- External GPU 1 activity elevated the first random 1M and head 128K/256K runs, including their all-local references. Those contexts were rerun. Initial raw files, fresh runs, and replacement provenance are retained; no samples were trimmed within a run.

## Paired optimization results (us)

| Layout | Context | Host | HBM reference | Staged | Segmented | Reduction vs staged | Latency increase vs HBM | Throughput loss vs HBM | Latency +20% target |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| random | 128K | 25% | 27.40 | 38.87 | 35.07 | 9.8% | 28.0% | 21.9% | Unmet |
| random | 128K | 30% | 27.40 | 43.63 | 40.67 | 6.8% | 48.4% | 32.6% | Unmet |
| random | 256K | 25% | 27.37 | 38.87 | 35.03 | 9.9% | 28.0% | 21.9% | Unmet |
| random | 256K | 30% | 27.37 | 43.13 | 40.20 | 6.8% | 46.9% | 31.9% | Unmet |
| random | 512K | 25% | 27.40 | 38.85 | 35.01 | 9.9% | 27.7% | 21.7% | Unmet |
| random | 512K | 30% | 27.40 | 43.17 | 40.22 | 6.8% | 46.8% | 31.9% | Unmet |
| random | 1M | 25% | 27.38 | 38.80 | 34.89 | 10.1% | 27.4% | 21.5% | Unmet |
| random | 1M | 30% | 27.38 | 43.13 | 40.06 | 7.1% | 46.3% | 31.7% | Unmet |
| head | 128K | 25% | 27.40 | 40.09 | 36.99 | 7.7% | 35.0% | 25.9% | Unmet |
| head | 128K | 30% | 27.40 | 43.91 | 40.95 | 6.7% | 49.5% | 33.1% | Unmet |
| head | 256K | 25% | 27.40 | 38.92 | 34.88 | 10.4% | 27.3% | 21.4% | Unmet |
| head | 256K | 30% | 27.40 | 43.35 | 40.03 | 7.7% | 46.1% | 31.5% | Unmet |
| head | 512K | 25% | 27.38 | 38.86 | 35.32 | 9.1% | 29.0% | 22.5% | Unmet |
| head | 512K | 30% | 27.38 | 43.46 | 41.01 | 5.6% | 49.8% | 33.2% | Unmet |
| head | 1M | 25% | 27.46 | 38.87 | 35.32 | 9.1% | 28.6% | 22.2% | Unmet |
| head | 1M | 30% | 27.46 | 43.30 | 41.13 | 5.0% | 49.8% | 33.2% | Unmet |
| tail | 128K | 25% | 27.39 | 38.88 | 34.95 | 10.1% | 27.6% | 21.6% | Unmet |
| tail | 128K | 30% | 27.39 | 43.21 | 40.33 | 6.7% | 47.2% | 32.1% | Unmet |
| tail | 256K | 25% | 27.46 | 38.93 | 34.93 | 10.3% | 27.2% | 21.4% | Unmet |
| tail | 256K | 30% | 27.46 | 43.38 | 40.05 | 7.7% | 45.8% | 31.4% | Unmet |
| tail | 512K | 25% | 27.38 | 38.92 | 34.90 | 10.3% | 27.4% | 21.5% | Unmet |
| tail | 512K | 30% | 27.38 | 43.37 | 40.08 | 7.6% | 46.4% | 31.7% | Unmet |
| tail | 1M | 25% | 27.43 | 38.89 | 34.89 | 10.3% | 27.2% | 21.4% | Unmet |
| tail | 1M | 30% | 27.43 | 43.52 | 40.14 | 7.8% | 46.3% | 31.6% | Unmet |

## Main comparison tables (us)

### Random

#### 128K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.46 | 27.40 | 27.69 |
| 5% | 44.63 | 30.15 | 27.79 |
| 10% | 50.51 | 32.01 | 27.74 |
| 15% | 56.09 | 34.03 | 27.72 |
| 20% | 60.62 | 37.70 | 30.98 |
| 25% | 65.54 | 42.09 | 35.07 |
| 30% | 69.89 | 46.60 | 40.67 |
| 35% | 73.70 | 50.98 | 45.32 |
| 40% | 78.37 | 55.36 | 49.27 |
| 60% | 94.49 | 72.76 | 71.19 |
| 80% | 111.96 | 90.60 | 89.44 |
| 100% | 129.11 | 108.03 | 106.40 |

#### 256K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.35 | 27.37 | 27.68 |
| 5% | 44.47 | 30.11 | 27.75 |
| 10% | 50.55 | 32.03 | 27.73 |
| 15% | 55.85 | 33.97 | 27.72 |
| 20% | 60.60 | 37.71 | 30.82 |
| 25% | 65.59 | 41.83 | 35.03 |
| 30% | 69.86 | 46.08 | 40.20 |
| 35% | 74.29 | 50.44 | 45.12 |
| 40% | 79.34 | 54.92 | 49.33 |
| 60% | 95.19 | 72.34 | 71.07 |
| 80% | 112.55 | 90.32 | 88.84 |
| 100% | 129.25 | 107.85 | 106.47 |

#### 512K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.44 | 27.40 | 27.69 |
| 5% | 44.31 | 30.11 | 27.74 |
| 10% | 50.95 | 31.72 | 27.72 |
| 15% | 55.51 | 33.76 | 27.73 |
| 20% | 62.02 | 37.48 | 30.88 |
| 25% | 65.90 | 41.75 | 35.01 |
| 30% | 70.85 | 46.07 | 40.22 |
| 35% | 74.39 | 50.45 | 45.30 |
| 40% | 78.31 | 54.78 | 49.31 |
| 60% | 94.79 | 72.39 | 70.77 |
| 80% | 111.98 | 90.49 | 88.87 |
| 100% | 129.72 | 107.36 | 105.85 |

#### 1M

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.44 | 27.38 | 27.68 |
| 5% | 44.12 | 30.10 | 27.72 |
| 10% | 50.39 | 31.76 | 27.71 |
| 15% | 56.16 | 33.72 | 27.71 |
| 20% | 60.66 | 37.34 | 30.69 |
| 25% | 65.67 | 41.66 | 34.89 |
| 30% | 69.75 | 46.02 | 40.06 |
| 35% | 74.66 | 50.40 | 44.92 |
| 40% | 77.35 | 54.75 | 49.21 |
| 60% | 94.01 | 72.23 | 70.71 |
| 80% | 109.30 | 89.69 | 88.12 |
| 100% | 127.10 | 107.16 | 105.53 |

### Head

#### 128K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.47 | 27.40 | 27.69 |
| 5% | 44.67 | 30.11 | 27.77 |
| 10% | 50.74 | 31.30 | 27.74 |
| 15% | 56.44 | 35.22 | 27.80 |
| 20% | 61.23 | 39.00 | 31.79 |
| 25% | 66.23 | 43.83 | 36.99 |
| 30% | 70.19 | 46.75 | 40.95 |
| 35% | 74.80 | 53.06 | 47.14 |
| 40% | 78.61 | 54.97 | 49.23 |
| 60% | 94.20 | 72.40 | 70.73 |
| 80% | 109.45 | 89.76 | 88.16 |
| 100% | 127.33 | 107.27 | 105.58 |

#### 256K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.36 | 27.40 | 27.68 |
| 5% | 44.33 | 29.69 | 27.71 |
| 10% | 50.21 | 30.79 | 27.71 |
| 15% | 54.78 | 33.96 | 27.71 |
| 20% | 60.11 | 38.18 | 30.69 |
| 25% | 65.31 | 41.83 | 34.88 |
| 30% | 68.78 | 46.04 | 40.03 |
| 35% | 73.58 | 50.40 | 44.79 |
| 40% | 77.99 | 54.67 | 49.21 |
| 60% | 93.42 | 72.14 | 70.71 |
| 80% | 109.83 | 89.61 | 88.12 |
| 100% | 127.12 | 107.11 | 105.53 |

#### 512K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.42 | 27.38 | 27.69 |
| 5% | 44.26 | 29.93 | 28.43 |
| 10% | 50.05 | 30.93 | 28.19 |
| 15% | 55.88 | 34.26 | 27.96 |
| 20% | 62.04 | 37.84 | 30.86 |
| 25% | 65.70 | 41.86 | 35.32 |
| 30% | 69.97 | 46.05 | 41.01 |
| 35% | 74.14 | 50.29 | 45.16 |
| 40% | 78.13 | 54.71 | 49.27 |
| 60% | 94.30 | 72.17 | 70.86 |
| 80% | 110.03 | 90.44 | 88.74 |
| 100% | 128.98 | 107.24 | 106.08 |

#### 1M

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.41 | 27.46 | 27.69 |
| 5% | 44.19 | 30.00 | 28.45 |
| 10% | 49.83 | 31.06 | 28.18 |
| 15% | 56.04 | 34.35 | 27.98 |
| 20% | 60.90 | 37.89 | 30.83 |
| 25% | 64.74 | 41.97 | 35.32 |
| 30% | 69.70 | 46.24 | 41.13 |
| 35% | 73.62 | 50.57 | 45.15 |
| 40% | 78.05 | 54.85 | 49.34 |
| 60% | 94.32 | 72.35 | 70.87 |
| 80% | 110.91 | 89.84 | 88.73 |
| 100% | 128.27 | 107.39 | 106.12 |

### Tail

#### 128K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.46 | 27.39 | 27.69 |
| 5% | 44.54 | 30.09 | 27.79 |
| 10% | 50.13 | 32.58 | 27.75 |
| 15% | 55.63 | 35.18 | 27.71 |
| 20% | 60.13 | 38.43 | 30.79 |
| 25% | 65.02 | 42.55 | 34.95 |
| 30% | 69.81 | 46.53 | 40.33 |
| 35% | 73.24 | 50.44 | 45.11 |
| 40% | 77.39 | 54.79 | 49.27 |
| 60% | 93.85 | 72.41 | 70.74 |
| 80% | 110.96 | 90.18 | 88.88 |
| 100% | 127.95 | 108.33 | 106.03 |

#### 256K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.42 | 27.46 | 27.69 |
| 5% | 44.66 | 30.22 | 27.75 |
| 10% | 50.10 | 32.73 | 27.73 |
| 15% | 54.86 | 35.49 | 27.72 |
| 20% | 59.17 | 38.37 | 30.73 |
| 25% | 65.23 | 42.55 | 34.93 |
| 30% | 68.31 | 46.82 | 40.05 |
| 35% | 73.40 | 50.76 | 45.06 |
| 40% | 76.73 | 54.91 | 49.38 |
| 60% | 93.49 | 72.18 | 70.88 |
| 80% | 110.57 | 90.38 | 88.15 |
| 100% | 127.86 | 108.14 | 106.56 |

#### 512K

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.31 | 27.38 | 27.69 |
| 5% | 44.44 | 30.05 | 27.75 |
| 10% | 49.93 | 32.51 | 27.72 |
| 15% | 55.93 | 35.50 | 27.71 |
| 20% | 61.85 | 38.47 | 30.74 |
| 25% | 64.90 | 42.57 | 34.90 |
| 30% | 70.05 | 46.61 | 40.08 |
| 35% | 74.17 | 50.51 | 45.11 |
| 40% | 77.17 | 54.92 | 49.32 |
| 60% | 94.33 | 72.51 | 70.73 |
| 80% | 110.79 | 89.86 | 88.76 |
| 100% | 128.65 | 107.30 | 106.23 |

#### 1M

| Host | Memcpy | Direct | Segmented |
|---:|---:|---:|---:|
| 0% | 27.44 | 27.43 | 27.69 |
| 5% | 43.99 | 30.00 | 27.77 |
| 10% | 50.40 | 32.49 | 27.74 |
| 15% | 56.18 | 35.29 | 27.71 |
| 20% | 60.69 | 38.33 | 30.71 |
| 25% | 65.24 | 42.55 | 34.89 |
| 30% | 69.92 | 46.61 | 40.14 |
| 35% | 73.71 | 50.38 | 44.98 |
| 40% | 77.56 | 54.84 | 49.23 |
| 60% | 93.41 | 72.16 | 70.76 |
| 80% | 110.17 | 89.70 | 88.20 |
| 100% | 127.94 | 107.20 | 106.26 |

## Verification

All 576 paired FP32 output comparisons passed; maximum absolute error 0.003749. Additional output/LSE, graph replay, and Host-mutation validation passed 372 cases. See [verification](verification.json) for API, sanitizer, traffic, resource, and source identities.

## Charts and data

Charts use the three main methods, context on the horizontal axis, and latency in us on the vertical axis. Error bars show the 10th-90th percentile across rounds. Existing baseline reports and plots are preserved.

[Charts](charts.zip) · [CSV](comparison.csv) · [Original comparison](../host_attention_comparison/report.md) · [Previous optimization](../host_attention_25_30/report.md)
