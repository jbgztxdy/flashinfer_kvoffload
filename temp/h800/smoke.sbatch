#!/usr/bin/env bash
#SBATCH --job-name=fi-host-smoke
#SBATCH --partition=h01
#SBATCH --nodelist=g40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/logs/%x-%j.out
# Smoke test on H800: JIT build, unit tests, then a small three-way comparison.
set -uo pipefail
source temp/h800/env_job.sh
OUT=temp/h800/results/smoke-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi -q | grep -A3 -i "PCIe Generation\|Link Width" | head -20
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvcc --version | tail -1
cp temp/h800/pydeps_jitpaths.py temp/h800/pydeps/jitpaths.py
python -m pytest -q -p no:cacheprovider -p jitpaths tests/attention/test_mla_host_batch.py 2>&1 | tail -15
python temp/bench_glm52_host_batch_kernel.py --device 0 --cpu-affinity "$(taskset -cp $$ | awk -F': ' '{print $2}')" \
  --index-mode mtp --requests 1 16 128 --fractions 0 0.1 0.3 --rounds 4 --index-sets 2 \
  --output "$OUT/mtp_small.json" 2>&1 | grep -v Warning | tail -20
echo "end=$(date -Is)"
