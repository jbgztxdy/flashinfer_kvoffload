# QK swizzled address cursors,2026-11-02

Retained the exact measured754706f8…ded79 header. Full1024-position3cold-pass reverse-order100GB0/20us gains1.492%/1.276%;50GB gain0.067%. Normal same-deviceHBM256x3gain0.183%;physicalPCIe0.005%essentially unchanged. This does not complete the overall objective.

The previous CKV paired-QK loop rebuilds the Q/K swizzle offset from row/column each dimension. The new loop initializes both once and uses the existing xor/add cursor update. Originalunroll4, fragments/load/MMA order, jointMDO/PV, QPEreuse, sourcehint, fetch/LRU/bytes/FAR/protocol/sharedlayout/launch remain. `make_candidate.py` checks16384address cases before building. `candidate.diff` isolates the source change; no rejected query-copy or cache-store mechanism was combined.

4focused BF16 tests cover longqueues128pools/capacity512/4096, sparseunequal pools/odd tails/empty rows, paddedQPEstride80, output/LSE/Graph/cachepayload. Each EMU and normal BF16 build keeps38original kernel machine/control words/resources identical; only2StreamSource entries change. Normal and EMU targetREG255/STACK16/LOCAL0/static2304B; no zero-stack claim. Actual Graph is still1kernel/108CTA/256threads/dynamic147968/active1.

Static IMAD/instruction counts increase despite fewer source expressions; `address_codegen_review.json` and SASS preserve this. Static counts are not dynamic cost, stalls or causal performance evidence. Timings decide retention.

16cells/8pairs, all same-parameter comparisons and two same-hardware references in `performance_audit.json`; raw retains all steps/passes/misses/hit rates/guards/SHA. GPU3/A100-PCIE/NUMA1, MTP8/R128/LRU4096,f24/splitoff; same-card far is not NVLink. No broadmatrix/sanitizer repetition, newFP16/SXM/NVLink/H800 or repeated productionroot compile/tests. `retention.json` and hashed `retention_snapshot/` bind exactly installed bytes and evidence.

Sources restore from frozenbase.cuh/qk_cursor.cuh and canonicalmath; includes/JIT ignored. Run scripts are archived for reproducibility; do not restart completed runs. `continuation_state.json` lists terminal handles. Negative independent candidates are in sibling cache_store/query_copy20261102 directories. Main report is docs/mla_host_a100_followup_20261102.md.
