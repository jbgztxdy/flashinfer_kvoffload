# Lane-private denominator, 2026-11-01

Rejected.100GB0/20us slower0.275/0.723%than frozenJQcontrol,50GBslower0.039%. No full/native/crossshape or reduction-interval/tile/unroll sweep. See [report](../../../docs/mla_host_a100_lane_denominator_20261101.md), decisions.json and performance_audit.json.

Same quantized BF16/FP16 P; eachlane accumulates its fourKV columns, finalquad shuffles once before native cross-warpgroup normalization. JointMDO factor isquad-uniform, real equation equivalent, FP32grouping differs. d starts1/4 to preserve emptyitemstate, oddtile uses same newlocal reduction. PV/barriers/refill/32KV2stage/fetch/LRU unchanged.109newtests include longLSE/empty/single/padding/Graph/allcachepayload,old34/68machinewords exact,actualone108CTAkernel,REG252BF16/253FP16STACK0. ReusedJQ109testevidence notnewtests.

9cells6basepairs3threewaypoints,1728timedlaunch108segments,bothreferences/source snapshots complete. All99321/1359/92142/22672terminal0,no failures or retries. SimplegroupedKload proposal already present in frozenJQSASS,not prototyped;qk_load_schedule_review.json preserves localstatic evidence,notperformance. Sourcegenerator/restore refuse overwrites. Root unchanged;originalJQretention1398independent,goalactive.
