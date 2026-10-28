#!/usr/bin/env bash
# A100 tuning: 100 GB/s + latency, and small batches, without changing cache capacity or traces.
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0
export PYTHONPATH=$PWD TMPDIR=$PWD/temp
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU
export FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu
dest=temp/sim_link/tune_100
mkdir -p "$dest"
run() {
  local name=$1
  shift
  if [[ -f $dest/$name.json ]]; then return; fi
  numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python "$@" --output "$dest/$name.json" \
    > "$dest/$name.log" 2>&1
}
for latency in 0 10000 20000; do
  if [[ ! -f $dest/validate_l$latency.json ]]; then
    numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/validate.py \
      100 "$latency" "$dest/validate_l$latency.json" > "$dest/validate_l$latency.log" 2>&1
  fi
  for fetch in 8 16 32 64; do
    modes=(ours_stage)
    if [[ $fetch == 16 ]]; then modes+=(ours_inplace memcpy_lru direct_lru); fi
    run mtp8_l${latency}_f$fetch temp/h800/bench_rounds.py --tier peer --mtp 8 \
      --requests 1 2 4 16 64 128 --passes 3 --fetch-ctas "$fetch" \
      --modes "${modes[@]}" --emu "100:$latency"
    run mtp1_l${latency}_f$fetch temp/h800/bench_rounds.py --tier peer --mtp 1 \
      --requests 1 4 16 128 --passes 3 --fetch-ctas "$fetch" \
      --modes "${modes[@]}" --emu "100:$latency"
  done
done
echo TUNING_DONE
