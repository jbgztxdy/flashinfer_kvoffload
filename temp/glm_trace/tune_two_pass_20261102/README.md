Frozen production754706f8 versus whole-item QK-then-PV,2026-11-02.

Rejected after2targeted BF16 output/LSE/Graph/full-valid-cache-payload tests and3paired256-position/3cold-pass points onGPU3/NUMA1. 100GB0/20us slower56.53/57.36%,50GB20us slower11.78%. No adoption or overall110%target claim. `decisions.json` binds final source and rawaudit; all runners terminal0.

`two_pass.cuh`, `two_pass.inc`, `workspace.py` contain the mechanism. Workspace is enlarged identically outside events for base and candidate. `base.cuh` and `base_mla.cuh` freeze baseline. Includes/JIT are ignored; restore by copying repo include into include_base/include_two_pass then overlaying the frozen corresponding host/math headers. Do not restart completed runners. `make_candidate.py` is original construction history, not an idempotent resume command.

`performance_audit.json`,gpu3/raw andgpu3/source_snapshots retain all samples/bytes/misses/guards/SHA andsame-hardware references. Do not mixGPU3 same-devicefarHBM withSXM NVLink. No full-suite/sanitizer/matrix repetition.
