Rejected independent schedule mechanism from production754706f8.

`pv_rescale.cuh` defers each existing joint-MDO O multiply to its feature's
first PV MMA. Same number of FMULs, same P and per-output MMA order, original
barriers, QK, fetch/FAR, exact launch LRU and real cache payloads. No geometry
or parameter sweep. `source_review.json` checks unrelated typed helpers.

Normal/EMU build: REG254 STACK0 SHARED2304 LOCAL0. Target SASS archived by
`review_schedule.py`; static instruction counts and operation proximity are
not dynamic stall or performance evidence. Two focused BF16 tests passed:
K65/K2048, padding/empty/odd, QPE stride80, output/LSE, Graph replay and all
valid cache payloads/mappings across3steps.

FreshGPU3/NUMA1 base→candidate, MTP8/R128/LRU4096/f24/off,256positions×3coldpasses:
100GB0/20us slower0.329/0.141%;50GB20us faster0.0136% (effectively flat).
No retained speedup, no expansion. Raw positive/negative samples, actual
miss bytes, two unchanged same-hardware references and guards retained.
This is same-card far-HBM emulation, not NVLink. No new fullsuite,
sanitizer, actualGraphnodeprobe, physical-link/FP16/SXM/H800 conclusion.

Build via `build.py`; already completed runs in `state.json`; do not rerun
`make_candidate.py` or the completed screen. Independent next experiment
`../tune_pv_feature_20261103` starts directly fromproduction, not this candidate.
