# Duplicate QK warpgroups, 2026-10-31

Rejected: normal105tests pass and all original32EMU/64normal library kernels unchanged, but new kernel uses255registers/192Bstack and100GB screens regress. Single cooperative kernel, native unsharded-QK math, unchanged cache/traffic/emulator/reference. [Report](../../../docs/mla_host_a100_qk_local_20261031.md).

Six256-position3pass cells/three same-parameter pairs, GPU3same-device far simulation. All raw, negative results, source snapshots, codegen archives and initial CLI/spill-gate failures remain. No full/native/global-retention claim. All GPU runners terminal; separate tune_qk_loop_20261031 tries bounded feature unrolling without overwriting this candidate.
