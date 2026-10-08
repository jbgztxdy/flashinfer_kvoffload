# Host attention: memcpy, direct access, and overlap

A100-SXM4-40GB GPU 1; BF16; one decode query; 64 heads; compressed rank 512; RoPE 64; top-k 2048; page size 1. A single attention layer is measured. Context labels are powers of two: 128K=131072, 256K=262144, 512K=524288, 1M=1048576.

Half of the physical cache resides in local HBM and half in pinned NUMA-0 Host memory. Hit fractions apply to the selected 2048 rows and are rounded to the nearest row. Random/head/tail describes their positions in the selected list; physical addresses are sampled across each tier.

## Measurement

- Memcpy: gather selected Host rows on CPU into a pinned buffer, then H2D DMA and native attention on one stream. Every repeated GPU cycle performs its own DMA. Local HBM is aliased directly, with no local gather or copy.
- Direct: native attention directly reads mapped Host memory.
- Overlap: the optimized fused attention kernel fetches Host rows concurrently with local computation.
- Main tables and charts report the median GPU cycle in µs. Memcpy includes DMA, its handoff/dependency cost, and attention. CPU packing is excluded from these GPU values and reported separately in details.md; serial wall time includes packing and synchronization.
- Each case uses the same KV data, query, and eight index sets across all three methods. Forty rounds are measured, with 100 serialized cycles per CUDA graph. Fraction and method order are shuffled. Error bars show the 10th-90th percentiles across rounds.
- Top-k selection, index updates/remapping, plan, initialization, and FP32 references are excluded. All measured cases passed independent FP32 comparisons.
- The 20% memcpy trace contains 100 actual pinned-Host-to-device copies and 100 native attention kernels on the same stream, with no overlap and no GPU gather kernels. Each copy moves 472320 bytes. See memcpy_verification.json for the checked timeline.

## Charts

The three main figures show random placement. Head and tail figures are in their corresponding subdirectories. PNG, SVG, and PDF versions are available.

![10% Host](host_10.png)

![15% Host](host_15.png)

![20% Host](host_20.png)

## GPU cycle latency (µs)

### Random

#### Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.45 | 27.42 | 27.35 | 27.48 |
| 5% | 44.24 | 43.56 | 43.82 | 43.97 |
| 10% | 49.84 | 49.31 | 49.88 | 49.91 |
| 15% | 55.43 | 52.67 | 55.71 | 55.26 |
| 20% | 59.79 | 58.24 | 59.53 | 61.61 |
| 25% | 64.53 | 64.01 | 65.11 | 64.95 |
| 30% | 68.61 | 67.91 | 70.11 | 68.40 |
| 35% | 73.07 | 70.88 | 74.14 | 73.66 |
| 40% | 76.72 | 75.87 | 77.79 | 78.43 |

#### Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.38 | 27.38 | 27.39 | 27.29 |
| 5% | 30.12 | 29.98 | 30.09 | 29.82 |
| 10% | 31.81 | 31.96 | 31.75 | 31.59 |
| 15% | 33.79 | 33.83 | 33.79 | 33.51 |
| 20% | 37.52 | 37.37 | 37.49 | 37.22 |
| 25% | 41.67 | 41.67 | 41.89 | 41.58 |
| 30% | 46.04 | 46.05 | 46.16 | 45.95 |
| 35% | 50.41 | 50.42 | 50.55 | 50.31 |
| 40% | 54.79 | 54.76 | 54.88 | 54.64 |

#### Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.69 | 27.71 | 27.71 | 27.71 |
| 5% | 28.74 | 28.75 | 28.77 | 28.76 |
| 10% | 28.74 | 28.74 | 28.74 | 28.74 |
| 15% | 28.74 | 28.74 | 28.74 | 28.74 |
| 20% | 32.76 | 32.63 | 32.68 | 32.52 |
| 25% | 40.93 | 40.94 | 40.96 | 40.92 |
| 30% | 45.22 | 45.36 | 45.33 | 45.26 |
| 35% | 50.08 | 50.11 | 50.08 | 50.05 |
| 40% | 55.11 | 55.09 | 55.17 | 55.09 |

### Head

#### Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.42 | 27.38 | 27.47 | 27.45 |
| 5% | 45.06 | 44.21 | 44.64 | 44.75 |
| 10% | 50.48 | 50.08 | 49.90 | 50.96 |
| 15% | 56.70 | 55.32 | 55.76 | 56.80 |
| 20% | 61.46 | 60.31 | 59.95 | 61.59 |
| 25% | 66.16 | 65.10 | 65.36 | 65.78 |
| 30% | 69.94 | 69.07 | 71.13 | 69.98 |
| 35% | 75.91 | 72.67 | 74.23 | 76.44 |
| 40% | 79.49 | 77.11 | 78.56 | 79.12 |

#### Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.36 | 27.36 | 27.39 | 27.45 |
| 5% | 30.11 | 29.72 | 30.00 | 30.34 |
| 10% | 31.29 | 30.83 | 30.97 | 31.39 |
| 15% | 36.31 | 34.02 | 34.34 | 34.93 |
| 20% | 39.65 | 38.04 | 38.85 | 39.20 |
| 25% | 43.32 | 41.85 | 42.14 | 43.89 |
| 30% | 46.88 | 46.14 | 46.33 | 48.75 |
| 35% | 50.80 | 50.31 | 50.54 | 52.52 |
| 40% | 54.90 | 54.55 | 54.72 | 54.97 |

#### Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.69 | 27.72 | 27.74 | 27.73 |
| 5% | 28.69 | 28.73 | 28.74 | 28.74 |
| 10% | 28.72 | 28.73 | 28.74 | 28.74 |
| 15% | 29.61 | 28.73 | 28.74 | 28.77 |
| 20% | 33.42 | 32.62 | 32.71 | 32.82 |
| 25% | 41.53 | 40.89 | 40.95 | 41.04 |
| 30% | 45.66 | 45.17 | 45.28 | 47.43 |
| 35% | 50.23 | 50.04 | 50.07 | 52.39 |
| 40% | 55.36 | 54.97 | 55.13 | 55.34 |

### Tail

#### Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.46 | 27.39 | 27.47 | 27.42 |
| 5% | 44.75 | 43.82 | 44.11 | 43.85 |
| 10% | 50.16 | 49.78 | 50.03 | 49.99 |
| 15% | 56.35 | 55.11 | 54.96 | 55.09 |
| 20% | 61.49 | 59.06 | 58.06 | 60.20 |
| 25% | 65.95 | 64.98 | 64.83 | 64.71 |
| 30% | 70.23 | 69.33 | 68.46 | 66.92 |
| 35% | 74.65 | 73.37 | 73.03 | 72.09 |
| 40% | 78.87 | 76.49 | 77.30 | 76.83 |

#### Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.39 | 27.36 | 27.42 | 27.43 |
| 5% | 30.03 | 29.78 | 29.85 | 29.69 |
| 10% | 32.90 | 32.26 | 32.29 | 32.17 |
| 15% | 36.03 | 34.94 | 34.91 | 34.80 |
| 20% | 39.35 | 37.91 | 37.98 | 38.00 |
| 25% | 42.66 | 42.10 | 42.08 | 42.14 |
| 30% | 46.96 | 46.25 | 46.27 | 46.27 |
| 35% | 50.50 | 50.27 | 50.26 | 50.30 |
| 40% | 54.75 | 54.61 | 54.59 | 54.66 |

#### Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 0% | 27.70 | 27.79 | 27.80 | 27.75 |
| 5% | 28.73 | 28.73 | 28.73 | 28.73 |
| 10% | 28.73 | 28.73 | 28.73 | 28.73 |
| 15% | 29.72 | 28.73 | 28.73 | 28.73 |
| 20% | 32.93 | 32.74 | 32.70 | 32.69 |
| 25% | 41.02 | 40.90 | 40.92 | 40.91 |
| 30% | 45.53 | 45.29 | 45.29 | 45.41 |
| 35% | 50.19 | 50.07 | 50.08 | 50.07 |
| 40% | 55.30 | 55.02 | 54.92 | 54.96 |

## Higher Host fractions (µs)

### Random: Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 93.21 | 92.99 | 94.28 | 93.27 |
| 80% | 110.02 | 109.89 | 111.07 | 109.81 |
| 100% | 128.12 | 127.72 | 128.36 | 127.15 |

### Random: Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 72.24 | 72.30 | 72.30 | 72.08 |
| 80% | 89.74 | 89.73 | 89.83 | 89.58 |
| 100% | 107.58 | 107.28 | 108.07 | 107.07 |

### Random: Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 75.93 | 75.93 | 75.92 | 75.87 |
| 80% | 97.33 | 97.36 | 97.45 | 97.31 |
| 100% | 118.00 | 117.89 | 119.16 | 117.62 |

### Head: Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 95.40 | 92.83 | 94.58 | 95.72 |
| 80% | 109.83 | 108.42 | 109.40 | 110.62 |
| 100% | 127.48 | 127.03 | 127.44 | 127.31 |

### Head: Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 72.31 | 72.03 | 72.18 | 72.33 |
| 80% | 89.81 | 89.57 | 89.64 | 89.83 |
| 100% | 107.25 | 107.05 | 107.16 | 107.30 |

### Head: Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 75.96 | 75.85 | 75.86 | 75.92 |
| 80% | 97.34 | 97.31 | 97.29 | 97.37 |
| 100% | 117.80 | 117.54 | 117.75 | 117.82 |

### Tail: Memcpy

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 94.83 | 92.84 | 92.86 | 92.59 |
| 80% | 109.53 | 108.70 | 108.46 | 108.07 |
| 100% | 127.31 | 127.04 | 127.01 | 126.94 |

### Tail: Direct

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 72.24 | 72.01 | 72.02 | 72.06 |
| 80% | 89.78 | 89.61 | 89.61 | 89.68 |
| 100% | 107.32 | 107.12 | 107.17 | 107.19 |

### Tail: Overlap

| Host | 128K | 256K | 512K | 1M |
|---:|---:|---:|---:|---:|
| 60% | 75.98 | 75.86 | 75.86 | 75.85 |
| 80% | 97.38 | 97.33 | 97.34 | 97.31 |
| 100% | 117.89 | 117.80 | 117.89 | 117.91 |

## Files

- [Component and serial-step tables](details.md)
- [Exploratory variants and legacy baseline tables](history.md)
- [CSV data](comparison.csv)

The old baseline artifacts are preserved. The new benchmark source is ../bench_glm52_host_comparison.py; raw JSON retains samples, source hashes, the Git commit, correctness errors, and GPU snapshots. GPU 1 shared the machine with unrelated workloads on other GPUs. This comparison does not measure a complete GLM model.

## Reproduce

```bash
PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/run_glm52_host_comparison.py
MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_comparison.py
```

Existing raw outputs are protected from overwrite. Choose new output names for a repeat run.
