#!/usr/bin/env bash
#SBATCH --partition=h01
#SBATCH --nodelist=g40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/logs/%x-%j.out
# Generic H800 job: sbatch --job-name=NAME --gres=gpu:h800:N --time=T --cpus-per-task=C --mem=M
#   temp/h800/job.sbatch [--tree DIR] COMMAND... ; COMMAND may use $OUT (per-job results dir).
set -uo pipefail
TREE=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer
if [ "${1:-}" = "--tree" ]; then TREE=$2; shift 2; fi
source /WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/env_job.sh
cd "$TREE"
export PYTHONPATH=$TREE:/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/pydeps
export OUT=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/results/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p "$OUT"
{
  echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES tree=$TREE start=$(date -Is)"
  echo "commit=$(git -C "$TREE" rev-parse HEAD) dirty=$(git -C "$TREE" status --porcelain include csrc flashinfer | wc -l)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,clocks.max.sm --format=csv
  nvidia-smi topo -m
  nvidia-smi -q -i 0 | grep -A2 -E "PCIe Generation|Link Width" | head -12
  python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
  nvcc --version | tail -1
  numactl -H 2>/dev/null | head -4
} 2>&1 | tee "$OUT/env.txt"
CPUS=$(taskset -cp $$ | awk -F': ' '{print $2}')
export CPUS
echo "cmd: $*"
eval "$@"
status=$?
echo "end=$(date -Is) status=$status"
exit $status
