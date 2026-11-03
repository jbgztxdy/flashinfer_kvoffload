# Native unlimited all-warp iteration — 2026-11-03

User priority: first beat current production with **normal builds, no emulator compiled**; revisit100GB modelling afterward. PhysicalGPU3/A100-PCIE, NUMA1, same-device HBM source. This is notNVLink. Control production is cab508dc, math34afad69, FAR8507eab0. Final accepted normal BF16 direct candidate is retained81093ba5; see retention.json for actual installation status. No H800/SXM tasks or other users' files touched.

Current final results are in `audit.json`; raw files retain every measured step, cold pass, actual unique misses, source/library SHA and hardware/exclusivity checks. Baseline is production normal JIT `../emu_fix_20261103/jit_v2_normal`. All candidates use private `include_*` and `jit_*`, and `FLASHINFER_EXTRA_CUDAFLAGS=''`.

All timed full runs are BF16, MTP8/R128/64heads/K2048/LRU4096, staged unless explicitly direct, host_split off,256positions/32steps,3coldpasses. CUDA Graph/event spans include one full cooperative MLA kernel. Planning/upload/validation outside events. Each new mechanism got two focused K65/K2048 cases with padding/empty/odd, QPEstride80, output/LSE, three cache traces, Graph twice, all valid cache payloads/mappings. No repeated full suite/sanitizers/matrix.

## Mechanisms and frozen sources

- `fused`: all108CTAs/full8mathwarps/dual32KVstages,2x10farbuffers. Original QK inner body restored, service phases absorbed into existing preQK/jointMDO/PV barriers. REG255/STACK8.
- `packet`: combine2x10farbuffers into1x20; same23KiB farstorage, halfpacketpublishes. REG255/STACK0.
- `release`: CTAbarrier→leader release→consumer acquire replaces redundant perwriter threadfence. `protocol.cu` separately checked38,880,000 uint4 payload vectors/54producer-consumer pairs/500epochs, errors0. Official ordering links in manifest.
- `phase`: separate active-service/pure-attention row bodies once thisCTA's own far jobs are done. REG255/STACK24.
- `rowcopy`: eight lanes per page, hoist source/cache/staging pointers out of nine vector rounds. REG255/STACK8. Fresh reverse pair recorded.
- `lean`: pure rows use original warp0 acquire followed by existing CTAbarrier, avoiding shared waiting broadcast. REG255/STACK16.
- `pvpipe`: distribute nextpacket cp.async rounds between PV feature MMA; no clear gain.
- `operands`: load both K/V operands before first MMA; no clear gain.
- `allready`: one acq_rel completion per producer CTA, acquire total+CTAbarrier before disabling tile-ready queries on later rows. About526/1024rows use it, but no gain.
- `qpetail`: reuse dead farshared for8KiB QPE after ownservice done, lower longlived registers in pure rows; slower.
- `stagger`: even/odd warp opposite-phase stores/QK and reads/PV, same8warps allcompute, normal-only prototype; no gain.
- `clean`: lean without nonfunctional perpacket diagnostic counters. All actual reads, staging/cache stores, LRU, publication, done tracking, math preserved. Counter fields unavailable, explicitly zeros; not proof of zero far traffic. Best staged prototype so far, still slower than production in reverse confirmation.
- `queue`: clean plus coarse dynamic row assignment; one initial counter gridbarrier; no gain. First CPU compile duplicate tid failure archived.
- `lrupipe`: start first realread before own distributedLRU, use specific poolready acquire before victim consumption, remove allLRU gridbarrier. No gain.
- `regstream`: normal-only32page packet copied through4temporaryregs, removes farshared, no persistent register payload. REG252/STACK0, dynamic140160. Slower despite lower resources. Actual source hint evict_first retained; no speculative prefetch or extra source reads.
- `native_diag`: **production diagnostic**, minimal phase timestamps/perrow count, no per-tile/fetchloop timers. Normal build; one32step pass, sampled math and allcache payloads checked. `native_timeline_analysis.json` reports timing and how many output rows fetchCTAs actually computed. Not an adopted optimization.
- `direct_pv`: native inplace prepare/LRU/cache-owner writeback + current pairedQK/QPEreuse/jointMDO/PVfeature math. Far bytes are per reference, not unique. First gate erroneously retained staged capacity condition and fell back to existing direct; archived failed_direct_gate. Corrected new path confirmed dispatch_probe_direct_pv.json and2focusedchecks;2856.469us, reverse2861.301us vs oldstaged3029.035us (5.538% faster). Olddirect fallback was3328.544us (14.186% faster arithmetic, differentrun). Actualread562.985MB/step; do notcallthisunique traffic. Normal only: do not use its unvalidated emulator dispatch.
- `trycache`: latest nonblocking direct same-launch cache reuse. Owner data stores→jointMDO CTAbarrier→pervictim release; nonowner may select ready cache or fall back to original far. No waiting for future owners. Actual selection counted per warp. Final actual version passed2focusedchecks/Graphprobe but3.701ms. Actualsource504.679MB/step vs direct562.985MB; reducedbytes did not improve latency. Rejected. Failed direct gate and preGPU perlane sourcechoice are archived.

Normal-only frozen `stagger`/`regstream` guard the new path; their unvalidated EMU fallback is not a production-equivalence claim. Rejected prototypes are not installed; retained normal BF16 direct math was installed after its final run, as recorded in retention.json. Do not use these prototypes for100GB timing.

The raw `setting.emulated=true` label inherited by early native harnesses is stale metadata; actual `config.normal_build=true`, `simulator_hooks=false`, build.ninja without `-DFLASHINFER_FAR_EMU`, and exact loaded normal library SHA determine execution. Later harnesses set emulated=false. The model header and device reference were never changed.

- `retained`: measured directPV math under normal-only guards and BF16 gate. EMU exclusion leaves original source tokens unchanged; staged/default route stays original. Actual REG254/STACK0/static1408/dynamic147968. `retained_scope_review.json`, finalrun and retention.json bind source and result.

## Reproduction

Frozen `.cuh` files are authoritative. Create a private include tree by copying current include/ then overwriting only flashinfer/attention/mla_host_batch.cuh with the selected frozen header; verify manifest SHA and canonical math/FAR SHA. Do not rerun one-time `make*.py` on existing files. First fused failed source/log in first_compile; queue/trycache failures have separate archives. Their build logs do not describe later fixed SHAs.

Compile at root using specified Python, CUDA_HOME=/usr/local/cuda-13.1, arch8.0, no extra flags, private workspace `jit_<variant>`, PYTHONPATH=root, TMPDIR=root/temp, `TUNE_VARIANT=<variant>`, `compile.py`. This is CPU-only; no GPU allocation.

`run_<variant>.py` is a fresh-run command, not idempotent resume: it checks GPU3 stable idle, uses NUMA1 and shared lock, focused checks then one native point. Existing output files are never overwritten by benchmark; do not restart completed runners. No queued/background waiters should remain on handoff. `audit.py` is read-only apart from its derived audit output and can be rerun.

The actual source bytes for staged are `unique_misses*1152`. `assist_stats` for base/clean/queue/lrupipe/regstream/direct_pv is unavailable; printed far_rows0 is **not** zero far traffic. DirectPV actual bytes use actual direct far references*1152; trycache counts actual selected sources. No new100GB/latency/physical/H800 conclusions or 110% completion claim.
