# Spatial fetch/QK feasibility probe — completed

See [report](../../../docs/mla_host_fetch_qk_assist_results_20261103.md), analysis.json andresult.json. 24/32CTA,4loader+4QKwarp,64rowdoublebuffer and12KiBlocalfeatureworkspace, unchangedv2header. 100GB0/20, three samples perpoint, alternatingmodeorder. Fullpayload, sampledcompleteQK andarrival>=deadline checks passed. Purefetch/loaderonly/assistREG32/32/70, stack/local0. No simultaneous84worker, LRU orreal scoreconsumer; all3361QKjobsfinishbeforefetch, so no saturatedassistcapacity claim.

pre_gpu_swizzle preserves plainsharedlayout CPUbuild, never GPUrun. FinalXORswizzles avoid native ldmatrix bankalignment conflicts; no pre/postperformance claim. Source andbinarySHAinresult. run.py is completed, no waiter; do not restart accepted probe. Fullintegration later showed negative end-to-end results and is not installed.
