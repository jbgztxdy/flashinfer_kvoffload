# query_copy,2026-11-02

Rejected focused screen. Mixed subpercent screen: -0.316%,+0.135%,-0.073%; no demonstrated useful gain.

Base is the exact retained48c2a724 production; private header SHA 9faf021939dbe6c2ffc2b82e8202ee55dd9698818f620ed316b2c0b58129f5df. `candidate.diff` isolates the change. `manifest.json`, `build_*.json`, `codegen.json`, `targeted_correctness.log`, `focused_state.json` and `performance_audit.json` bind the actual source and outcomes. Independent includes/JIT are ignored and reproducible from the frozen headers. No existing production header was overwritten.

BF16-only independent EMU/normal builds: all38 original instances in each build retain identical machine/control words/resources; only the existing StreamSource target changes. No FP16 code-identity claim. Shared GPU3 lock and resource pre/post checks used; same-card far HBM is not NVLink. All6 raw cells/3pairs,1152timed launches/72valid segments and both same-hardware references retained, including all negatives. No full suite, sanitizer repetition or broad matrix. All runners ended.

`make_candidate.py` creates the source snapshot and independent include trees; `run_builds.py` builds, `run_focused.py` checks and benchmarks, `check_codegen.py` archives code identity, `audit_performance.py` verifies timings and references. Do not restart completed runs. Overall objective incomplete.
