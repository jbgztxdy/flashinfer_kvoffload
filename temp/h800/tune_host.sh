# H800 PCIe tuning: fetch-CTA count for the staged path, and in place (all SMs read Host) vs staged.
set -x
for F in 8 16 32 64; do
  python temp/h800/bench_rounds.py --tier host --mtp 8 --requests 16 64 128 --passes 3 --fetch-ctas $F --modes ours_stage memcpy_lru floor_fa2 --output $OUT/rounds_mtp8_f$F.json 2>&1 | grep steady
  python temp/h800/bench_rounds.py --tier host --mtp 1 --requests 64 512 --passes 3 --fetch-ctas $F --modes ours_stage memcpy_lru floor_fa2 --output $OUT/rounds_mtp1_f$F.json 2>&1 | grep steady
done
python temp/h800/bench_rounds.py --tier host --mtp 8 --requests 16 64 128 --passes 3 --modes ours_inplace hisparse --output $OUT/rounds_mtp8_inplace.json 2>&1 | grep steady
python temp/h800/bench_rounds.py --tier host --mtp 1 --requests 64 512 --passes 3 --modes ours_inplace hisparse --output $OUT/rounds_mtp1_inplace.json 2>&1 | grep steady
