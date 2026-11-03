Sharedhashpre-dedupcandidate,2026-11-02. Production754706f8unchanged.

2026-11-03 final update: freshGPU3samehardware3pairedpoints completed;100GB0/20us slower5.72/6.65%,50GB20us slower3.70%. Rejected; do notresume oldSXMtimings. `decisions_gpu3_20261103.json` andcurrentcontinuation_state override the historical pending instructions below.

Finalcandidatea47ca8b8 normal/EMU BF16builds and2targetedSXM correctnesschecks passed. `prepare.inc` explains4same-pool-rowhash beforeglobalCAS. Originalattention/fetchliteral checks andscratchbounds in `source_review.json`. Pre-GPUscratchalias draft archivedinpre_gpu_alias; do notrunold7420draft.

ThreefreshSXMbaselinecells completed. Firstcandidatecell hitforeignGPU0/1processes atfinalresourcecheck; rawcomplete=false andexcluded. Its3440us isnotvalidperformanceevidence. Nocandidategain/regressiondecisionyet. `focused_sxm_state.json`, `continuation_state.json`, logs andincompleterawpreservefailure. Noactiverunner/waiter, noforeignprocesssignaled.

Resumeonlywhenresourcesareidle: run `resume_candidate.py` withrepositoryPython. ItfirstpreflightsGPU0/1, thenbenchacquiresgpu01lockandarchivesexistingincompleterawbefore3missingcandidatetimings. Itdoesnotrerunbaseorcorrectness. Thenrun `audit_performance.py`; retainonlywithvalidsame-parameterdata. Includes/JITignored; reconstructfromrepoincludeoverlaidwithbase/shared_claimandcanonicalmlafrozenheadersifneeded. `make_candidate.py`/`fix_scratch.py`areconstructionhistory, notresumecommands.
