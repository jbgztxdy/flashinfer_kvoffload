# Validation on H800: unit tests, then tiny sweeps and a tiny HiSparse comparison.
set -x
cp temp/h800/pydeps_jitpaths.py temp/h800/pydeps/jitpaths.py
python -m pytest -q -p no:cacheprovider -p jitpaths tests/attention/test_mla_host_batch.py 2>&1 | tail -5
python temp/h800/bench_sweep.py --tier host --requests 1 16 --fractions 0 0.3 --rounds 3 --near-requests 16 --output $OUT/sweep_host_tiny.json 2>&1 | grep -v Warn | tail -8
python -c "from sgl_kernel.flash_mla import flash_mla_sparse_fwd; from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla; print('hisparse imports ok')"
python temp/h800/bench_rounds.py --tier host --requests 1 4 --mtp 1 --steps 8 --passes 2 --near-requests 4 --output $OUT/rounds_host_tiny.json 2>&1 | grep -v Warn | tail -8
python temp/h800/bench_rounds.py --tier host --requests 2 --mtp 8 --steps 4 --passes 2 --near-requests 2 --output $OUT/rounds_host_mtp8_tiny.json 2>&1 | grep -v Warn | tail -5
