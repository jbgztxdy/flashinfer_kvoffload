# Joint probability publication, 2026-11-01

Rejected:100GB0/20us slower0.182/1.115%than exactJQcontrol;50GB+0.048%. No full/native/crossshape expansion or synchronization/tile/unroll/fetch/cadence scan. See [report](../../../docs/mla_host_a100_joint_pv_20261101.md), decisions.json, performance_audit.json and composition_comparisons.json.

Original2stage/32KV/108CTA/fetch/LRU unchanged; existing joint-MDO barrier protects KPE->P overwrite, publish both P with one barrier, native descending PV/refill. Four PV barriers become one, odd tile native. Canonical math untouched.109normaltests,34EMU/68normal instructions/resources/machinewords exact,REG252STACK0LOCAL0,actualone108CTA/block32x4x2kernel; separate synccheck0errors,racecheckfirstmatchinglaunch0hazards are diagnostic,notperformance/exhaustiveproof. ReusedJQ109testevidence isnotnewtests.

9cells6basepairs3threewaypoints,1728timedlaunch108segments,bothreferences/raw/source snapshots complete. First sanitizerattachtimeout and missingsourceplans.cpp failure preserved; onlyfailedshortscreen retried, passedgates not repeated. Allrunnersterminal. Originalvalidation_state.json records firstscreenfailure; successful recovery59627 and appendedscreen state preserved. Sourcegenerator/restore refuse different frozen inputs. Root untouched by this candidate; JQretention1398independent,overallgoalactive.
