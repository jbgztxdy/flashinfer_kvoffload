#!/usr/bin/env bash
# Local A100 integration check only; the formal 64K/1024 traces are collected separately.
set -euo pipefail
cd /mnt/data/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export FLASHINFER_CUDA_ARCH_LIST=8.0
export FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final PYTHONPATH=$PWD
dest=temp/glm_trace/results
trace=temp/glm_trace/validation8k/doc0.pt
test -f "$trace"
run() {
  local name=$1
  shift
  if [[ -f $dest/$name.json ]]; then return; fi
  numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python "$@" \
    --captures "$trace" --context 8192 --output "$dest/$name.json" > "$dest/$name.log" 2>&1
}
run a100_8k_smoke_host_mtp1 temp/h800/bench_rounds.py --tier host --mtp 1 \
  --requests 1 4 --steps 256 --passes 3 \
  --modes ours_stage ours_inplace memcpy_lru direct_lru floor_fa2
run a100_8k_smoke_peer_mtp8 temp/h800/bench_rounds.py --tier peer --mtp 8 \
  --requests 1 4 16 --steps 32 --passes 3 \
  --modes ours_stage ours_inplace memcpy_lru direct_lru floor_fa2
run a100_8k_smoke_peer_mtp8_hbm temp/h800/bench_sweep.py --tier peer --mtp 8 \
  --requests 1 4 16 --fractions 0 --rounds 8 --modes attn direct batch
python3 temp/glm_trace/06_report_replay.py "$dest/a100_8k_smoke_host_mtp1.json" \
  "$dest/a100_8k_smoke_peer_mtp8.json" --hbm-probes "$dest/a100_8k_smoke_peer_mtp8_hbm.json" \
  --out "$dest/a100_8k_smoke.md"
