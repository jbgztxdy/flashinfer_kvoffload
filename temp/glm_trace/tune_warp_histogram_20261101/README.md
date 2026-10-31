# Warp-private small-batch LRU histograms

The frozen base is JQ a2f37fb9, not production b6. `manifest.json` records both
source hashes and the exact dispatch/mechanism. Production is not changed by
these scripts. Existing JQ confirmation/retention processes remain authoritative.

`make_candidate.py` creates the frozen pair and isolated include trees.
`restore_includes.py` restores missing trees without overwriting existing ones.
Private include and JIT trees are ignored by Git; measured headers, math,
source snapshots, raw timings, logs and SASS/resources are retained.

The original handles started in this experiment are:

- 16107: normal BF16/FP16 CPU compile then 105 correctness tests on exclusive GPU3.
- 31851: emulated BF16 CPU compile.
- 84049: wait for normal correctness, then 4 exact-radix/cache/Graph tests.
- 24248: wait for the original producers, compare 36/72 old instances including
  machine/control words, prove actual single-kernel Graph dispatch, run
  synccheck/racecheck, then a fixed two-shape three-setting screen and audit.

Do not duplicate a live handle or treat a tool timeout as process failure.
GPU3 is A100-PCIE NUMA1; same-card independent HBM is not NVLink.
All GPU work uses the shared GPU3 correctness lock and resource observations.

Initial performance plan: MTP1/R1 and MTP8/R1, f16/host_split=on,
256 positions ×3 cold passes, JQ(base) → warp_hist,
100GB0µs / 100GB20µs / 50GB20µs. `short_plan.json`, `validation_state.json`,
`performance_audit.json`, `decisions.json` and the generated report record the
actual completion state and any later validation. No parameter sweep is used.

`test_histogram_lru.py` also injects widely separated launch stamps to exercise
all four radix bytes. It computes the precise expected victim slot set using
the existing thread-major tie traversal, verifies protected hits and every
cached payload, then checks zero-miss Graph replay. This validates the changed
integer aggregation mechanism in the actual one-kernel path.

Overall optimization goal remains active regardless of this candidate's outcome.

Final state: all five original handles above and the additional forced-radix
sanitizer handle **69911** are terminal exit0. 109 normal tests passed, 36/72
original JQ machine/control words matched, and all four sanitizer invocations
reported zero errors/hazards. The additional forced-radix probe instruments the
second matching launch because the first cold-cache launch may skip radix.
No failed attempts or retries occurred.

The 12 cells /6 pairs contain 10368 timed launches and648 valid timing segments.
All same-hardware compute and max references are present. Gains range from
-1.047% to+0.111%; the candidate is rejected with no full/native/cross-shape
extension. Do not repeat a histogram-row, warp-group or merge-layout sweep.
The original JQ72075/1398 processes remain live and wait for the last three
SXM forward cells; they alone control possible JQ production retention.
