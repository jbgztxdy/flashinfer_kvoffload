# Audit of the unchanged far-link emulator (2026-11-03)

User asked whether100GB/s emulation is exact and where latency is added.
Production kernel cab508dc and simulator5e409035 remain unchanged. No timing,
reference, old calibration or raw benchmark was rewritten. This is a correctness
review of the simulator, not a kernel optimization or a newperformance result.

The intended virtual server is S_i=max(S_(i-1),issue_i)+floor(bytes_i/B), with
ready_i=S_i+L. L is excluded from S to allow pipelined requests. The actual
fetch issues cp.async, records issue time, waits for real-copy completion, then
calls reserve/waits for the virtual deadline before staging writes and ready
publication. Therefore latency is relative to virtual transmission completion,
not an unconditional sleep L after real-copy completion. Actual publication
also includes staging, barriers and scheduling overhead. Reservation happens
at consume time, using the recorded issue time; queue order is reservation
order rather than a strict sort of issue timestamps.

Confirmed defect in reserve: atomicAdd(clock,cost) followed by conditional
atomicMax(clock,issue+cost) is not one linearizable max-plus update. Multiple
requests restoring an idle clock can lose charged bytes. Two simultaneous
73728B requests at100GB/s each cost737 integerns. Legal sequence A.add,B.add,
A.max,B.max gives both the deadline issue+737 instead of issue+737 andissue+1474.
`reserve_interleavings.json` enumerates the legal schedules; counts are not
probabilities or measured frequency.

`reserve_probe.cu` includes the unchanged production emulator header and calls
its actual reserve helper. Cooperative24CTA/32threads; synchronized synthetic
same-issue-time requests, no payload or waits/timing measurement. Eachcase2000
trials on exclusiveGPU3/NUMA1 under sharedlock. Idle2CTA:2000clock mismatches;
idle24CTA:2000mismatches, service spans737..8844ns versus17688ns expected.
Busy2CTA andbusy24CTA:0mismatches, exact spans1474/17688ns. Output
`reserve_probe.json` binds source/emulatorSHA and resource observations. These
constructed cases demonstrate correctness failure, not its frequency/effect
in the full MLA trace. The probe's clock reset is confined to its separate
executable/device module; production simulator/config and header were not edited.

Prior `kernel_report_20261029/calibration.json` is SXM0/1,16row/432CTA standalone
gather: saturated100GB/s measured99.72..99.96GB/s over1.2GB; dependent20us access
measured20.526us. It validates the long saturated average for that harness;
it does not establish transient reservations,24CTA/64row production path,
GPU3 residency fidelity, or true hardware100GB/s equivalence. The gather and
fetch use the same reserve helper but different hook/timestamp locations.

Even absent the defect, same-card HBM simulation still consumes localHBM/L2
for far-source reads and uses SM-resident polling; it is not physicalNVLink.
Existing AB raw remains evidence within the unchanged emulator, but exact100GB
physical-path equivalence and causes of the~24%gap remain unproven. Quantify
production idle-recovery incidence before attributing an end-to-end impact.

Recommended next step: independent/versioned simulator correctness candidate
with atomic max-plus reservation and idle/burst/saturated cases; separately
measure emulator overhead and production fetch timeline, then collect fresh
paired timings/references as needed. Do not overwrite or silently relabel old
results. No simulator fix or alternate-reference experiment was run here.

Build diagnostic:
/usr/local/cuda-13.1/bin/nvcc -O3 -arch=sm_80 -std=c++17 -I include reserve_probe.cu -o reserve_probe
Run on exclusiveGPU3 with sharedlock,CVD3 andnumactl-N1-m1. No taskprocessleft.
