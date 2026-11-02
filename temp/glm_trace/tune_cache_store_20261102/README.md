# cache_store,2026-11-02

Rejected focused screen. No positive screen point; 100GB20us regresses0.908%.

Base is the exact retained48c2a724 production; private header SHA 0805d7c0f4a8e96e09ffa9d0a3c1eb4a44a541fb21bb2e6ee848ffe4bd5acbba. `candidate.diff` isolates the change. `manifest.json`, `build_*.json`, `codegen.json`, `targeted_correctness.log`, `focused_state.json` and `performance_audit.json` bind the actual source and outcomes. Independent includes/JIT are ignored and reproducible from the frozen headers. No existing production header was overwritten.

BF16-only independent EMU/normal builds: all38 original instances in each build retain identical machine/control words/resources; only the existing StreamSource target changes. No FP16 code-identity claim. Shared GPU3 lock and resource pre/post checks used; same-card far HBM is not NVLink. All6 raw cells/3pairs,1152timed launches/72valid segments and both same-hardware references retained, including all negatives. No full suite, sanitizer repetition or broad matrix. All runners ended.

`make_candidate.py` creates the source snapshot and independent include trees; `run_builds.py` builds, `run_focused.py` checks and benchmarks, `check_codegen.py` archives code identity, `audit_performance.py` verifies timings and references. Do not restart completed runs. Overall objective incomplete.
