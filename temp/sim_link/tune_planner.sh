#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0
export PYTHONPATH=$PWD TMPDIR=$PWD/temp
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu
dest=temp/sim_link/tune_100
run() {
  local name=$1
  shift
  test ! -f "$dest/$name.json"
  numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/bench_tune.py "$@" \
    --output "$dest/$name.json" > "$dest/$name.log" 2>&1
}
for fetch in 24 48; do
  run mtp8_l20000_f$fetch --tier peer --mtp 8 --requests 1 2 4 16 64 128 \
    --passes 3 --fetch-ctas "$fetch" --modes ours_stage --emu 100:20000
done
for latency in 0 20000; do
  for split in 0 1; do
    run small_mtp8_l${latency}_split$split --tier peer --mtp 8 --requests 1 2 4 \
      --passes 3 --fetch-ctas 16 --modes ours_stage ours_inplace --emu "100:$latency" \
      --host-split "$split" --host-item-entries 256
    run small_mtp1_l${latency}_split$split --tier peer --mtp 1 --requests 1 4 16 \
      --passes 3 --fetch-ctas 16 --modes ours_stage ours_inplace --emu "100:$latency" \
      --host-split "$split" --host-item-entries 256
  done
done
echo PLANNER_DONE
