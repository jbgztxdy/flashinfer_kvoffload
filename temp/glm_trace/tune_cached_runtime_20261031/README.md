# Runtime cache predicate with cached-only shared IDs

Only restore the original runtime `pool_capacity != 0` predicate in the cached-only shared-ID candidate. Dispatch still uses old kernels for noLRU/generic/whole controls. No math/cache-policy/emulator/far-byte change, no production install.

Normal105tests pass. All original32EMU/64normal library kernels are exact instructions/resources, including pure-device math. Newcached target REG244/STACK0/LOCAL0; `predecessor_comparison.json` proves its machine instructions/resources equal the first unrestrictedfetch_ids cached target in both builds/dtypes. That original candidate's noLRU regression is not reintroduced. Code identity is not a fresh performance/causation proof.

Fresh independent confirmation is complete:26cells/13pairs,including20full1024-position5pass cells acrossSXMlimited/native andGPU3limited/native,plus6short-screen cells. SXM100GB0/20us is-0.653%/-0.122%,normalNVLink+1.841%,PCIe-0.085%;GPU3full100GB0-0.458%,20+0.378%,50+0.308%,normalHBM+1.395%,PCIe-0.290%. No stable100GB improvement or established advantage over cached_ids; not adopted. No broad cross-shape matrix has been run for this source. All four confirmation handles are terminal0.

`allocation_probe.json`/`allocation_audit.json` add a same-allocation diagnosis onSXM:three distinct capturedCUDAfunctions,equal launch shape,each5full cold passes at100GB0/20us. All30passes check output, mapping and all524288actualcacheKVrows. At0us cached_ids-0.272%/runtime_ids+0.044%;at20us bothabout+0.6%. This is not matrix replacement or a causal explanation for every cross-run variation. Probe handle72917terminal0.

W's preceding cached-ID matrix/full confirmations proceed independently without overwritten inputs. Its640limited cells and120GPU3native cells are complete; normalSXM136 remains in progress under originalhandle92282. Its full5pass100GB+0.3% differs from several three-pass negative cells; all remain. No production change; historical32-34%large100GBgap unresolved. [Report](../../../docs/mla_host_a100_cached_runtime_20261031.md).

Rebuild ignored include trees on a fresh checkout with `restore_includes.py --round cached_runtime` (also supports wait/heads/qk/qk_loop). It verifies the unchanged simulator and frozen host/math sources; existing trees are checked, never overwritten. JIT libraries can then be rebuilt with compile.py under the recorded environment; baseline builds are shared with H.
