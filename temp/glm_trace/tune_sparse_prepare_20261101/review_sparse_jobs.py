"""Structural oracle for bounded pool ranges, dense job lookup and ready flag identity."""
from bisect import bisect_right
import hashlib
import itertools
import json
from pathlib import Path
import random

p = Path(__file__).resolve().parent
rng = random.Random(20261105)
cases = []
for pools, length, fetch_ctas, mode in itertools.product([1, 4, 128], [256, 2048], [4, 24], range(6)):
    row_counts = [rng.randint(1, 8) for _ in range(pools)]
    starts = [0]
    for n in row_counts:
        starts.append(starts[-1] + n)
    counts = [0 if mode == 0 or (mode == 1 and i % 3 == 0) else
              min(row_counts[i] * length, [1, 15, 63, 97, 513, 2047][mode]) for i in range(pools)]
    unique = sum(counts)
    rows = 16 if unique < (3 if fetch_ctas <= 8 else 1) * fetch_ctas * 64 else 64
    prefix = [0]
    for n in counts:
        prefix.append(prefix[-1] + (n + rows - 1) // rows)
    physical = set()
    flags = set()
    issued_rows = 0
    for job in range(prefix[-1]):
        # Independent upper-bound oracle, checked against the kernel binary search.
        expected_pool = bisect_right(prefix, job) - 1
        first, last = 0, pools
        while first + 1 < last:
            mid = (first + last) // 2
            if prefix[mid] <= job:
                first = mid
            else:
                last = mid
        assert first == expected_pool
        offset = (job - prefix[first]) * rows
        base = starts[first] * length + offset
        count = min(rows, counts[first] - offset)
        assert 0 < count <= rows
        assert base % rows == 0 and base + count <= starts[first + 1] * length
        flag = base // rows
        assert flag not in flags
        flags.add(flag)
        assert flag < (starts[-1] * length + 15) // 16
        for slot in range(base, base + count):
            assert slot not in physical
            physical.add(slot)
            assert slot // rows == flag
        issued_rows += count
    assert issued_rows == unique == len(physical)
    expected = {starts[pool] * length + i for pool in range(pools) for i in range(counts[pool])}
    assert physical == expected
    cases.append(dict(pools=pools, length=length, fetch_ctas=fetch_ctas, mode=mode, unique=unique,
                      fetch_rows=rows, jobs=prefix[-1], total_staging_capacity=starts[-1] * length))
record = dict(complete=True, cpu_only=True, cases=cases, case_count=len(cases),
              candidate_sha256=hashlib.sha256((p / 'sparse_prepare.cuh').read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limitation='Supplied pool counts only; not GPU liveness/ordering, LRU selection, math or performance proof.')
(p / 'sparse_job_review.json').write_text(json.dumps(record, indent=2) + '\n')
print('SPARSE_JOBS_OK', len(cases))
