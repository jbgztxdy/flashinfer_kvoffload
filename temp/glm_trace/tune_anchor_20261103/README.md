Rejected exponent-anchor mechanism,2026-11-03. Production754706f8unchanged.

Finalac18beac keeps a rounded-up base2anchor to skipstableOrescaling. Three targeted BF16tests pass, including large/logit-rebase andstrictLSEchecks. Firsta0a07draft failedsingle-elementLSE; frozeninfailed_lse. FinalusesseparateFP32denominatorforrequestedLSE, no tolerancechange. SamequantizedPdenominatorstillnormalizesoutput.

FreshGPU3/NUMA1,256positions x3coldpasses,MTP8/R128/LRU4096,f24off:100GB0us faster0.292%,20us slower1.674%,50GB20us flat(-0.003%). No stablegain;rejected. `decisions.json` bindsthefullrawaudit/SHA. Noactiverunner,noinstalledcandidate,nofullsuite/sanitizerrepeat. Do notrestartcompletedrun_focused.

Includes/JITignored; restorefromrepoincludeoverlaidwithfrozenbase/anchorandcanonicalmath ifneeded. make_candidate/fix_lse areconstructionhistory, notidempotentresumecommands.
