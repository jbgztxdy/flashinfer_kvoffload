# Far source eviction hint

The candidate changes only dedicated far-source cp.async loads. The source is deduplicated and attention uses the resulting staging/cache rows. Neither the transferred bytes nor readiness/fence/commit/wait protocol changes. Whether the cache hint helps requires actual code generation and matched timings.

NVIDIA PTX ISA §9.7.10.28.3.1 documents cp.async with an L2 cache policy operand; §9.7.9.16 documents createpolicy, including fractional evict-first priority. They are performance hints and do not replace synchronization. These forms are supported on sm80; the private build targets sm80. Policy1.0 is fixed for the streaming-source hypothesis, not a parameter sweep.

Source: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). Documentation checked 2026-11-01. No claims of measured hit rates or isolated stall reduction follow from this instruction choice.
