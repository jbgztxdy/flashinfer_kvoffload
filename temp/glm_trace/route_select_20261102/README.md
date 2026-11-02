# Current-source matched route selection

All planned30cells/15pairs complete. CurrentCUDA source754706f8 and originalnormal/EMU libraries are unchanged. SXM0/1 NV12/NUMA0, MTP8, BF16/K2048/h64. Sharedgpu01lock and resourcepre/post checks respected. GPU2/3 not used. `plan.json` defines twoLRUshapes16/128 and one16request noLRU shape at0/.1/.3, normalNVLink and100GB0/20us, bothroutes. AllLRUfull1024positions3coldpasses; noLRUfouroriginalsnapshots3passes.

`audit.py` verifies source/bindings/samplemeans/guards/coldmisses and originalsame-hardware stepwise compute/max references; `selected_routes.json` chooses the mean-minimizing route per matched key without deleting slower data. `select_route.configure_wrapper` applies that measured profile before warmup/capture; unmeasured configurations are explicitly unsupported. Not an automatic inference of physical transport, not a CUDA change or universal optimum.

`selected_correctness.json` covers bothchosenLRUpaths on shortK97fixtures, repeatedGraph/output/LSE/allvalidcachepayloads. Initialpytestimportfailure preserved, repaired by extracting twoexistinghelperdefinitions withast without dependency installation. No fullsuite/sanitizer/matrix repetition.

`history_review.json` identifies the oldNVLink7percent16request result and lowfar100GBresults, including oldersteady/fastest-sample reference differences. Do not conflate those with128requestLRU or currentallstepmeans.

Reproduce with `run.py` (skips complete raw per benchmark); read-only audits via `audit.py` and `report.py`. Inspect actualresource/state first; allcurrentrunners are terminal. `continuation_state.json` records finalstatus. Production remains754706f8. Full report: docs/mla_host_route_selection_20261102.md.
