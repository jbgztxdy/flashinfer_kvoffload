#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace/logs/%x-%j.out
# CPU-only gate; exit 0 means saved data and loss meet the handoff criteria.
set -euo pipefail
cd /WORK/PUBLIC/luyouy_work/rtx
source env.sh >/dev/null 2>&1
cd yjw/test/glm_trace
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
python 05_inspect_trace.py "$@"
echo "end=$(date -Is)"
