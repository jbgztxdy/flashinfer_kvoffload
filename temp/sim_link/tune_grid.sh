#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0
export PYTHONPATH=$PWD TMPDIR=$PWD/temp
export FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU -DFLASHINFER_HOST_BATCH_TUNE'
export FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu_tune
dest=temp/sim_link/tune_100
for grid in 32 64 108; do
  for mtp in 1 8; do
    requests=(1 4 16)
    if [[ $mtp == 8 ]]; then requests=(1 2 4); fi
    name=grid${grid}_mtp${mtp}_l20000_f16
    test ! -f "$dest/$name.json"
    numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/bench_tune.py \
      --tier peer --mtp "$mtp" --requests "${requests[@]}" --passes 5 --fetch-ctas 16 \
      --modes ours_stage ours_inplace --emu 100:20000 --grid-ctas "$grid" \
      --output "$dest/$name.json" > "$dest/$name.log" 2>&1
  done
done
echo GRID_DONE
