# Full H800 NVLink set at the final kernel (mirrors temp/a100_full/run.sh, host tier).
set -x
R8="1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80 96 112 128"
R1="1 2 4 8 16 32 64 128 256 512"
F="0 0.05 0.1 0.15 0.2 0.25 0.3"
python temp/h800/bench_sweep.py --tier peer --mtp 8 --requests $R8 --fractions $F --rounds 6 --modes attn copy direct batch --output $OUT/sweep_peer_mtp8.json 2>&1 | grep -v Warn | tail -3
python temp/h800/bench_sweep.py --tier peer --mtp 1 --requests $R1 --fractions $F --rounds 6 --modes attn copy direct batch --output $OUT/sweep_peer_mtp1.json 2>&1 | grep -v Warn | tail -3
python temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 2 4 8 12 16 24 32 48 64 96 128 --passes 5 --modes ours_inplace ours_nocache memcpy_lru direct_lru floor_fa2 --output $OUT/rounds_peer_mtp8.json 2>&1 | grep -v Warn | tail -3
python temp/h800/bench_rounds.py --tier peer --mtp 1 --requests 1 2 4 8 16 32 64 128 256 512 --passes 5 --modes ours_inplace ours_nocache memcpy_lru direct_lru floor_fa2 --output $OUT/rounds_peer_mtp1.json 2>&1 | grep -v Warn | tail -3
