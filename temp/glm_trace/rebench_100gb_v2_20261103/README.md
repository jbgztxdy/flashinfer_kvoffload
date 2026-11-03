# Fresh100GB/s measurement after emulatorv2 fix

Completed onexclusiveGPU3/A100-PCIE,NUMA1,same-cardfarHBM. NotNVLink.
Attentioncab508dc/math34af/FAR8507eab0,unchanged afterv2fix. BF16,h64,
KV512+64,K2048,MTP8/R128,LRU4096/request,f24/item64/splitoff,staged,
256positions=32steps,3coldpasses,allstep/pass arithmeticmeans.

`bench.py` is a frozen copy of the existing timing/input harness. Only changed:
- Newdirectory/rawprefix, modelv2 andMemcpypathv3 metadata, exact resolved v2
  include path to reuse previously validated v2binary SHAs.
- FailfastGPUpreflight/nonblocking sharedlock;source/binaryprovenance records.
- Memcpy gather usescurrentv2 emulator; its subsequent local-onlyattention is
  explicitly bound to the previously built NORMAL native device library.
  Sameplan/ABI/math,outputchecksretained;noemulationhooks onlocalcompute.
Thisbaselineprotocol isdocumented; historicalv1MemcpytotalsarenotstrictAB.

NoGPUkernel/timing/math/referenceformula modification. Newnormaldevice
reference usesexactsame32stepinput/bindings,notoldtimes ormax(mean).
Actualloadedlibrarypaths andSHAs checked in eachraw; privateincludes remain
at../emu_fix_20261103/include_v2,linksignored. Frozenhost/math/FAR/emu sources
andmanifest support reconstructionwithoutoverwritingoldJIT.

100GB0us:ours3327.371us,Memcpy5813.493us,device2558.933us,
stepwisemax2701.022us,+23.189%,saving42.765%vsMemcpy.
100GB20us:ours3360.555us,Memcpy5824.800us,device2558.933us,
stepwisemax2701.192us,+24.410%,saving42.306%vsMemcpy.
Actualunique247.915/247.957MBperstep,referencehit76.737/76.734%.
110%targets2971.124/2971.311us;remainingreduction10.707/11.583%.

24compute-dominated/8transfer-dominatedsteps inboth. Meanexcess0us:
compute611.285us,transfer671.539us;20us:639.801/718.049us.
Coldstepours6792.533/6816.085us vs6210.835us reference,+9.366/9.745%.
Wholemeanexcess626.349/659.363us. Latencysettingmean delta33.184us(~1%);
passrangesoverlap,notastatisticalorstrictcausallatencyeffectclaim.
84CTAroughcapacity:2558.933*108/84=3290.057us. Interestinghypothesis,
notlinear-scalingproof,notaphaseprofile. ActualEMUtargetREG254/STACK0/
static2304/LOCAL0. NoSM/gridgeometrychange.

480timedlaunches/30acceptedsegments,3newrawfiles,allnumerical/guard/resource/
source/binaryauditspassed. Noadditionalfullsuite/sanitizer/full1024matrix,
physicalNVLink/PCIe/H800,newmechanismorcausalprofile. Productionunchanged.
AfirstattemptwasstoppedPRE-CUDA:externalflashcheckrestartedbetweenparent
observationandchildpreflight;preflight_failure preservesit. Finalrunwaited
10secondsstableidle;allacceptedstart/endhadnoforeignprocess. Allrunnersended.

Readstate.json,audit.json,analysis.json. `audit.py` verifiesrawandprovenance.
Do notreruncompletedrun.py oroldrunners toinspectresults. Nohiddenwaiter.
