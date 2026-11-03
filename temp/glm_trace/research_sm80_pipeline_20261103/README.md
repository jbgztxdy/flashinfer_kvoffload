# SM80 same-warp pipeline research — compile/CPU evidence only

[Report](../../../docs/mla_host_sm80_pipeline_research_20261103.md) and `feasibility.json` explain limits. sources.json links official fixed-tag CUTLASS/FlashAttention code read through web tool; external snapshot download failed SSL, no downloaded source files or source hashes exist.

`probe.cu` independently implements a schedule microprobe: all8warps load local Q/K +perform trueQK, while issuing separate far64rowdoublebuffer copies, either bulk or18copy rounds across36MMA steps. Originalv2 FAR header unchanged. Compare24/108CTA100GB0/20. Same localoperandmovement acrosscontrols; no O/softmax/actual LRU/workerremap or necessaryQKreplacement, no directearlyfarconsumption. CPUcompile onlyREG117/168/76,stack/local0. No finalSASSinstructionoverlapinspection. No performance claim.

`layout.cu` compiled and executed on CPU (hostonlysizeof, no CUDAdevice API); layout.json confirms139776B compactattention+23040B remoteservice=162816dynamic. Assumed2304static total165120B fits166912B knownblocklimit. Not realkernel resources/occupancy or QPE-removal correctness validation. Sourceaudit isolates currentPVFeature numericalregion: native load_q writes sharedQPE, helper reads PEpackedregisters directly; canonicalotherpaths require sharedQPE.

run.py attemptedstable10sGPU3preflight under sharedlock; externalVLLM occupied physicalGPU3,120stimeoutbeforeCUDA/probechild. run_state.json hasobservations,finalstatus. **No result.json, allGPUcorrectness/performance untested**, no waiter. Originalkernel/FAR/device reference untouched. Do not restart old completed cofetch/QKassist experiments.

Next justified hardwarewindow: verify current production/FAR andprobe SHA inmanifest before running only this newrun.py. It refuses existingresult. numactlNUMA1,CUDA_VISIBLE_DEVICES3. Source/binary compiledCUDA13.1SM80, binaryignored. If absent rebuild:

```
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_80 -I include -Xptxas=-v temp/glm_trace/research_sm80_pipeline_20261103/probe.cu -o temp/glm_trace/research_sm80_pipeline_20261103/probe
/mnt/data/mhy-vllm-venv/bin/python -u temp/glm_trace/research_sm80_pipeline_20261103/run.py
```

Even positiveprobe would only justify further design. Entirekernel must preserve exactunique farbytes,LRU/cachepayload andfullattention; allwarpsparticipating cannot beused as adoptioncriterion. Newfulluniformkernel is NOT implemented.
