set -x
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=/home/rtx/projects/flashinfer/temp CUDA_VISIBLE_DEVICES=0,1
for i in 1 2 3; do
  cd /home/rtx/projects/flashinfer_base && PYTHONPATH=/home/rtx/projects/flashinfer_base FLASHINFER_WORKSPACE_BASE=/home/rtx/projects/flashinfer_base/temp/jit_ab2 numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/h800/bench_sweep.py --tier host --mtp 8 --requests 64 128 --fractions 0 0.1 0.3 --modes attn batch --rounds 8 --no-check --output /home/rtx/projects/flashinfer/temp/a100_final/after_pools/ab2_base_$i.json
  cd /home/rtx/projects/flashinfer && PYTHONPATH=/home/rtx/projects/flashinfer FLASHINFER_WORKSPACE_BASE=/home/rtx/projects/flashinfer/temp/jit_a100_final numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/h800/bench_sweep.py --tier host --mtp 8 --requests 64 128 --fractions 0 0.1 0.3 --modes attn batch --rounds 8 --no-check --output /home/rtx/projects/flashinfer/temp/a100_final/after_pools/ab2_new_$i.json
done
