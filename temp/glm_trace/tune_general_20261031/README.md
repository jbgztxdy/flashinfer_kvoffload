# A100 general optimization, 2026-10-31

Continue the current whole-row baseline (`base.cuh`, SHA256 `81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b`). Its implementation and the previous round's completed evidence were preserved in commit `5157ff83`; `verify_previous.py` re-runs that round's read-only sample/source audit without regenerating its original outputs.

Current status is in `audit_general.json`, `decisions.json`, and `continuation_state.json`. The copied `experiment.json` is a settings registry from the original matrix; its old `status` does not describe this optimization round.

- `profile.cuh` / three `raw/profile_*.json`: completed, instrumented, full-window single-pass diagnostics. `audit_general.py` independently decodes all 128 epochs × 108 CTAs. Instrumentation has a fixed 128-step/128-CTA capacity and is not suitable for MTP1 or a larger grid without changes.
- `warp_ready.cuh`: per-warp staging acquisition using the page IDs already loaded. Four 256-position, 3-pass cells and paired controls completed. It was slower at 100 GB/s and NVLink; not adopted.
- `row_ready.cuh`: acquire the entire row before entering the tile loop. Four 256-position, 3-pass cells and paired controls completed, slower on all four links; not adopted. Emulated and normal builds compiled, but normal GPU correctness was not claimed.
- `defer_cache.cuh`: keep staging publication immediate; move cache binding/scatter out of the fetch pipeline and re-read HBM staging afterward. Adds HBM traffic and cache-fill tail. Normal-build 103 tests passed, but four same-parameter SXM screen settings regress 29–35%; rejected.

`raw/` keeps every per-step/per-pass timing, LRU hit/miss counts, mathematical checks, queue guards, GPU process snapshots and executed source SHA. `source_snapshots/` freezes the actual executed source. `compiler.py` compares all six generic SM80 BF16 kernels and saves compressed SASS; candidate whole-row code is separately listed. The emulator, cache policy and original 5304-cell matrix are unchanged; the retained host dispatch is documented below.

Use the existing Slurm jobs and watchers; do not submit duplicates or overwrite their source. The resumed read-only snapshot is `h800_queue_resume_20261031.txt`. No new H800 measurements exist.

To reconstruct ignored include trees, use `python .../restore_include.py <variant>`; it refuses to overwrite a different existing tree. Run each candidate with its own `FLASHINFER_WORKSPACE_BASE=.../jit_<variant>`. `compile.py` supports CPU compilation with `CUDA_VISIBLE_DEVICES=''`. `run_readiness.py` serializes local GPU experiments, checks GPU0/1 availability and uses the shared experiment lock inside the benchmark. Do not launch a second runner while an existing handle is live.

The two completed screening commands were:

```bash
/mnt/data/mhy-vllm-venv/bin/python -u temp/glm_trace/tune_general_20261031/run_readiness.py
/mnt/data/mhy-vllm-venv/bin/python -u temp/glm_trace/tune_general_20261031/run_readiness.py --variants row_ready base --tag screenrow256_
```

The original first-screen driver is frozen as `run_readiness_screen_v1.py`. The current third screen is:

```bash
/mnt/data/mhy-vllm-venv/bin/python -u temp/glm_trace/tune_general_20261031/run_readiness.py --variants defer_cache base --tag screencache16_ --fetch-ctas 16
```

Do not use a short-window result as a full-trace claim. A promising candidate still needs identical-parameter full 1024-position forward/reverse cold-start passes, both references, cross-scenario regression and full mathematical/cache/Graph tests. `audit_general.py` verifies completed data only and never marks the overall objective complete.

## Later continuation: native regression and shape correction

The uniform `threshold2` / `threshold2_sm80` proposal is rejected. Despite complete limited-link/LRU gains, normal-build no-LRU 224-row cases regress by 3.2–4.4%. Native data is not interchangeable with an emulation build whose runtime hooks are disabled. `promote_medium.py` is intentionally disabled; that proposal was never installed.

`balanced_cached_sm80.cuh` compares the original split plan with direct whole rows using fixed tile costs and the existing cyclic work assignment, and memoizes the five scalar shape inputs in one bounded thread-local entry. Both balanced variants passed 105 normal-build tests; 360 cache-equivalence checks and normal/emulated compiler comparisons are saved. Final source SHA is in `manifest_balanced_cached_sm80.json`.

Final GPU3 native 80-cell validation is complete. The original runner rejected one partial setting after a foreign process entered; its preserved partial is in `gpu3/rejected/`, and the restart-safe runner completed only missing work. Final SXM normal NVLink/PCIe five-pass results are in root `raw/balanced_sxm_native_*`. GPU3 data lives in `gpu3/raw/`; `audit_gpu3.py` always uses GPU3-specific native references/calibrations and explicitly separates same-device HBM from remote links. The original matrix is unchanged.

CPU shape-model estimates are scheduling heuristics, not changes to any benchmark overlap reference. Normal-build tests are separate records per candidate. Boundary-only predecessor records are preserved in `test_history/`; the 105-test retained-shape test source is archived by SHA in `source_snapshots/`. `retain_balanced.py --install` passed all gates and installed the exact tested header (SHA52919…ff2c5). Final validation has 102 paired cells: all 82 changed-path pairs improve, and the 20 unchanged-path pairs have at most 0.31% negative difference. Frozen gate inputs and all negative differences are retained. See [retention report](../../../docs/mla_host_a100_balanced_dispatch_20261031.md). The overall goal remains active; the large-batch gap is unchanged.
