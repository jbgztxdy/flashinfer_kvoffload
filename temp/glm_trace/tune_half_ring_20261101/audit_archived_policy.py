"""Keep the first cohort with its actual one-buffer policy and corrected diagnosis."""
import hashlib,json,math,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
review=read(p/'invalid_budget_review.json');old=read(p/'invalid_budget/performance_audit.json')
files=[];rows=[]
for item in review['original_raw']:
    f=root/item['archive_path'];assert sha(f)==item['sha256']
    d=read(f);assert d['complete'];cfg=d['configuration']
    for key,h in d['source_sha256'].items():assert sha(p/'gpu3/source_snapshots'/(h+Path(key).suffix))==h
    assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
    for case in d['cases']:
        for mode,t in case['timings'].items():
            samples=t['per_step_samples_us'];assert len(samples)==32 and all(len(x)==3 for x in samples)
            assert all(math.isfinite(x) and x>0 for xs in samples for x in xs)
            assert abs(statistics.mean(x for xs in samples for x in xs)-t['mean_us'])<1e-7
    files.append(dict(path=item['archive_path'],sha256=sha(f),variant=cfg['variant']))
for row in old['rows']:
    row=dict(row);row['raw']=str((p/'invalid_budget/raw'/Path(row['raw']).name).relative_to(root));rows.append(row)
assert len(files)==len(rows)==9
record=dict(complete=True,adoption_eligible=False,category='Unintended one-fetch-buffer control; original negative half-ring measurements retained',
    out_of_bounds_claim_retracted=True,first_hypothesis_archive='invalid_budget/initial_incorrect_bounds_hypothesis.json',
    storage_policy=dict(no_copy_pe=dict(storage_bytes=139904,fetch_buffers=1,dynamic_shared_bytes=139904),
                        base=dict(storage_bytes=147968,fetch_buffers=2,dynamic_shared_bytes=147968),
                        half_ring=dict(storage_bytes=158336,fetch_buffers=2,dynamic_shared_bytes=158336)),
    device_memcheck=dict(log='memcheck_one_buffer_control.log',sha256=sha(p/'memcheck_one_buffer_control.log'),
        api_errors_reporting='disabled after preserving first diagnostic25errors (printed16 cuGetProcAddress_v2 API errors,9suppressed); device memory checking retained',
        kernel_scope='first matched NoCopyPEKernel launch, BF16 K97',zero_errors='ERROR SUMMARY: 0 errors' in (p/'memcheck_one_buffer_control.log').read_text()),
    timed_launches=old['timed_launches'],accepted_segments=old['accepted_segments'],rows=rows,files=files,
    tests=old['tests'],stress_tests=old['stress_tests'],goal_complete=False,candidate_adopted=False)
assert record['device_memcheck']['zero_errors']
(p/'archived_policy_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print('ARCHIVED_POLICY_VERIFIED',len(rows),record['timed_launches'])
