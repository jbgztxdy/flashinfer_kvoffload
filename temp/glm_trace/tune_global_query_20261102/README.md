Frozen production754706f8 versusglobal-query64KV/two-stage attention,2026-11-02.

Rejected after2targeted BF16 output/LSE/Graph/full-valid-cache-payload tests and3paired256-position/3cold-pass points onGPU3/NUMA1. 100GB0/20us slower44.45/43.10%,50GB20us slower6.72%. Production unchanged. `decisions.json` andperformance_audit bind evidence. No110%target claim; all runners terminal0.

`attention.inc` andglobal_query.cuh hold final33062352. Pre-GPU coordinate draft is inpre_gpu_layout; noGPUrun ofold draft. Includes/JIT ignored. Restore from repo include plusfrozenhost/math headers. Originalmake/fix scripts preserve constructionhistory; do not reruncompletedexperiments. `run_focused.py` used the alreadybuilt frozen7547baseline from sibling two_pass/jit_base, same normal/EMUflags.
