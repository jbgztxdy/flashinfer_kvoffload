# Bounded duplicate-QK unrolling, 2026-10-31

Rejected. Feature unroll4 for the newfalse-QK_SHARD/CTA32 type removes the previous192Bstack (nowREG244,STACK0,LOCAL0), but the three paired100/50GB screens remain slower.105normal tests pass; all original32EMU/64normal library kernels have exactly equal instructions/resources, including pure-device attention. [Report](../../../docs/mla_host_a100_qk_loop_20261031.md).

Private host/math headers, independentJIT, originalfailedsource and all raw/snapshots/codegen retained. Screens are256positions3coldpasses onGPU3same-device far simulation, notNVLink. No full/native/retention claim. No tuning unroll2/8 without a new mechanism. Auditor requires normaloriginalkernel equivalence before reusing same-hardware compute references despite changedprivateMathSHA. All GPU runners terminal; production remains52919.
