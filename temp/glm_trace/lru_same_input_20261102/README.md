# LRU same-input accounting,2026-11-02

User questioned batch scaling and LRU-on/off comparison. Existing lowfar noLRU cases (10/30percent initialremote) are not toggles of fulltraceLRU (nearly100percent initialremote). `existing_accounting.json` derives both traffic and idealreference from exactly the same fulltrace; it has no capacity0 GPU timing.

128requests/MTP8: unique672.920MB/step withoutcache vs259.582MB withcache; referencehit78.945percent, uniquehit61.425percent. At100GB, idealreference6.729ms withoutcache vs2.691ms withcache. Actualcachedtime3.290ms is already below the no-cache transferreference. This does not measure totalcachebenefit or attribute all remaining0.598ms toLRU.

16to128requests: compute0.503to2.467ms (4.9x), cacheduniquebytes32.44to259.58MB (8x). Morebatch createsmorework but notproportional sparecompute capacity. At100GB0/20us, measuredrelativegap actuallydecreases from34.6/40.4percent to22.2/22.9percent. NVLink selectedroutegap increases10.9to22.3percent; resourcecompetition/cachemaintenance/computeefficiency are mechanisms to investigate, not a measuredtimebreakdown.

Prepared twoactual capacity0stagedfulltracepoints (NVLink,100GB0) inrun.py/bench.py. Both useexistingcurrent source/binaries and Graph/event timing; CAP=0 alone disablescache. GPU0/1foreignflashcheckjobsoccupied atresourcepreflight. OwnCPUwaiter289478 stoppedbeforeopeningCUDA; parentexit1 preservesintentionalstop. NoGPUmeasurements/results, noforeignsignal. `continuation_state.json` is authoritative: noactive runner. Do notclaim theseplannedtimings exist. Only resumeoncardsfree if furthercausalcomparisonneeded. Currentkernel754706f8 unchanged.
