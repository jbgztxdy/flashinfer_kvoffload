#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0
export PYTHONPATH=$PWD TMPDIR=$PWD/temp
export FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU -DFLASHINFER_HOST_BATCH_TUNE'
export FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu_lru_tune
dest=temp/sim_link/tune_lru
for spec in '1 32' '1 48' '1 64' '8 6' '8 10' '8 12'; do
  read -r mtp chunks <<< "$spec"
  requests=(1 4)
  if [[ $mtp == 8 ]]; then requests=(1 2 4); fi
  name=chunks${chunks}_mtp${mtp}_l0
  test ! -f "$dest/$name.json"
  numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/bench_tune.py \
    --tier peer --mtp "$mtp" --requests "${requests[@]}" --passes 5 --fetch-ctas 16 \
    --modes ours_stage ours_inplace --emu 100:0 --kv-chunks "$chunks" \
    --output "$dest/$name.json" > "$dest/$name.log" 2>&1
done
echo FINE_SCAN_DONE
