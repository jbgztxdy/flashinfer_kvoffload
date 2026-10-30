# Measurement corrections retained for reproducibility

- Initial CPU scalar upload to diagnostics could wait on the GPU guard. Diagnostics now allocate zeroed storage before the guard; no scalar host upload occurs in the launch queue. Failed guard jobs are excluded.
- The legacy one-query prefetch path lacks complete virtual-link hooks. All reported shapes explicitly use the batched staged kernel with 16 fetch CTAs and 64-entry items. Early one-query measurements were archived.
- Formal local references were remeasured sequentially on GPUs 0/1 after smoke validation finished. Old references are archived.
- Data seed is reset per setting so Q/KV values are identical across physical and virtual links for a given batch/MTP. Preliminary comparisons before this correction are archived.
- LRU unique misses are averaged across three passes; all per-step miss samples are retained. Reference-weighted misses are read from the actual staged remap after the timed kernel event.
- Host DMA source buffers are distinct per queued step, so later CPU packing cannot overwrite an in-flight transfer.
- Input metadata, packing, CPU planning and output checks are excluded from GPU event intervals. Memcpy includes H2D or peer gather, local cache scatter and complete attention.
- No kernel implementation or default production dispatch was changed by these benchmark fixes.
- No-LRU copy baseline version 2 writes its contiguous local staging output directly, removing a redundant local scatter. Verified kernel samples are preserved with their source SHA; only affected copy baselines are remeasured. Prior copy timings and files remain archived.
- Later LRU measurement queues 16-step CUDA Graphs once with external per-step timing events and post-event diagnostics. Capture does not execute. Complete-window cross-check agrees within 0.2% for ours and 1.8% for Memcpy. An exclusive GPU benchmark lock prevents concurrent processes.
