# Native64KV/one-stage allocation-matched control, 2026-11-01

Rejected; production b6eea095…e72a /0f82e9de, goal active. See [report](../../../docs/mla_host_a100_kv64_20261101.md), decisions.json and performance_audit.json.

Original64head/256thread math uses native sharded64KV/1stage with the same147968B dynamic shared as32KV/2stage. One online softmax update per64KV; FP32 grouping changes, canonical math header untouched. Original fetch64row double buffer/f24,metadata,LRU and payload retained. Distinct from1stage/32KV and paired-QK-only controls, no parameter sweep. All109tests pass;34EMU/68normal old instances exactly equal. Actual one108CTA/block32x4x2 kernel,active1,static2304B. NewREG255/STACK8/LOCAL0, actualSTL/LDL in SASS; no stack-zero claim.

Six cells/three pairs,1152 timed launches/72segments and both references complete. At100GB0/20us slow1.576%/1.795%;50GB+20us fast0.024% (effectively flat), preserved. No adoption/full-window/native-link/cross-shape expansion. All runners terminal. restore_includes.py validates or restores frozen host/math without overwrite; baselines reuse W libraries. Source snapshots/raw/test/codegen/actual-dispatch and audit records retained; original5304matrix untouched.

The separate joint_mdo experiment retains32KV/2stage but combines two tiles' maximum/output rescale, refilling one stage during the other PV. Its independent gates and handles are in its own directory; no benefit inferred from this rejected control.
