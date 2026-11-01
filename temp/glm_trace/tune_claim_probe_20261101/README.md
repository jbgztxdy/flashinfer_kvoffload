# Claim atomics experiment

Both candidates are rejected. Production remains JQ7193. Claim probe regressed in the short screen. Claim store's full reverse improvement did not reproduce in full forward order; the extension parent exited 1 at its predefined positive-100GB gate. Do not restart it or run the unexecuted matrix plans.

The direct base is current production, not the pending cached-PE candidate. `manifest.json` and `store_manifest.json` bind the two candidate headers. `memory_model_review.md`, `atomic_source_review.json`, the prepare diffs, and complete SASS archives document the mechanism. `reused_production_base.json` binds the existing production baseline binaries and validation; their tests are not counted as new.

`performance_audit.json` and `results.csv` preserve 28 rows / 14 pairs, all timings, references, raw-file hashes, and 226 unique tests. `gpu3/raw/` contains per-step/per-pass samples, hit/miss counts and queue guards. GPU3 independent far HBM is not NVLink. The original 5304-row matrix is untouched.

Recreate missing private include trees with `restore_includes.py`; it refuses to replace differing existing trees. `compile.py`, `compiler.py`, `compiler_store.py`, the two test drivers, stress drivers, probes and screens record the build and execution environment. The corrected test fixture and all failed fixture/source/validator records are preserved; see `test_fixture_repair.json` and `test_fixture_failure/`.

CPU audits: run `audit_performance.py`, `review_atomics.py`, `checkpoint_audit.py`, then `report.py` using the task Python. The checkpoint deliberately distinguishes finished rejected experiments from unfinished broad regression. See [the Markdown report](../../../docs/mla_host_a100_claim_atomics_20261101.md).
