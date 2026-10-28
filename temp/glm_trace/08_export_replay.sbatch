#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace/logs/%x-%j.out
set -euo pipefail
cd /WORK/PUBLIC/luyouy_work/rtx
source env.sh >/dev/null 2>&1
cd yjw/test/glm_trace
python 08_export_replay.py "$@"
