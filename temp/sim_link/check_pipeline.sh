#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0
export PYTHONPATH=$PWD TMPDIR=$PWD/temp
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu
dest=temp/sim_link/tune_100
for latency in 0 20000; do
  for mtp in 8 1; do
    requests=(1 2 4 16 64 128)
    if [[ $mtp == 1 ]]; then requests=(1 4 16 128); fi
    name=pipeline_inline_mtp${mtp}_l${latency}_f16
    test ! -f "$dest/$name.json"
    numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/bench_tune.py \
      --tier peer --mtp "$mtp" --requests "${requests[@]}" --passes 3 --fetch-ctas 16 \
      --modes ours_stage ours_inplace --emu "100:$latency" --output "$dest/$name.json" \
      > "$dest/$name.log" 2>&1
  done
done
echo PIPELINE_DONE
