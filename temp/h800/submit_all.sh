# Queue the H800 measurement jobs (run on the login node from the repo root).
set -e
J=temp/h800/job.sbatch
R1="1 2 3 4 6 8 12 16 24 32 48 64 80 96 112 128"
M1="1 2 4 8 16 32 64 96 128 192 256 384 512 768 1024"
sub() { sbatch --parsable "$@"; }
V=$(sub --job-name=fi-validate --gres=gpu:h800:1 --cpus-per-task=8 --mem=96G --time=00:40:00 $J bash temp/h800/validate.sh)
echo validate=$V
sub --job-name=fi-ab --gres=gpu:h800:1 --cpus-per-task=8 --mem=96G --time=00:45:00 $J bash temp/h800/ab.sh
sub --job-name=fi-sweep-host-mtp8 --gres=gpu:h800:1 --cpus-per-task=16 --mem=96G --time=00:50:00 $J \
  python temp/h800/bench_sweep.py --tier host --mtp 8 --requests $R1 --output '$OUT/sweep.json'
sub --job-name=fi-sweep-peer-mtp8 --gres=gpu:h800:2 --cpus-per-task=16 --mem=96G --time=00:50:00 $J \
  python temp/h800/bench_sweep.py --tier peer --mtp 8 --requests $R1 --modes attn copy direct batch batch:f24 batch_inplace --output '$OUT/sweep.json'
sub --job-name=fi-sweep-host-mtp1 --gres=gpu:h800:1 --cpus-per-task=16 --mem=200G --time=00:50:00 $J \
  python temp/h800/bench_sweep.py --tier host --mtp 1 --requests $M1 --output '$OUT/sweep.json'
sub --job-name=fi-sweep-peer-mtp1 --gres=gpu:h800:2 --cpus-per-task=16 --mem=96G --time=00:50:00 $J \
  python temp/h800/bench_sweep.py --tier peer --mtp 1 --requests 1 2 4 8 16 32 64 96 128 192 256 384 512 --output '$OUT/sweep.json'
sub --job-name=fi-rounds-host-mtp1 --gres=gpu:h800:1 --cpus-per-task=16 --mem=160G --time=00:50:00 $J \
  python temp/h800/bench_rounds.py --tier host --mtp 1 --requests 1 2 4 8 16 32 64 128 256 512 --passes 5 --output '$OUT/rounds.json'
sub --job-name=fi-rounds-host-mtp8 --gres=gpu:h800:1 --cpus-per-task=16 --mem=96G --time=00:45:00 $J \
  python temp/h800/bench_rounds.py --tier host --mtp 8 --requests 1 2 4 8 16 32 64 128 --passes 7 --output '$OUT/rounds.json'
sub --job-name=fi-rounds-peer --gres=gpu:h800:2 --cpus-per-task=16 --mem=160G --time=00:50:00 $J \
  "python temp/h800/bench_rounds.py --tier peer --mtp 1 --requests 1 2 4 8 16 32 64 128 256 --passes 5 --output \$OUT/rounds_mtp1.json && python temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 2 4 8 16 32 64 128 --passes 7 --output \$OUT/rounds_mtp8.json"
squeue -u $USER -o "%i %j %T %l %R"
