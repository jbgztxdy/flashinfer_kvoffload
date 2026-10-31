"""Compare the isolated cached target with the first shared-ID kernel, not its dispatch."""
import hashlib
import json
from pathlib import Path
p=Path(__file__).resolve().parent;w=p.parent/'tune_wait_20261031'
load=lambda f:json.loads(f.read_text())
records=[]
for normal in [False,True]:
    if normal:
        previous=load(w/'compiler_normal_fetch_ids_summary.json')
        targets=[r for r in previous['records']['candidate'] if 'WholeRowHostBatchParams' in r['symbol'] and 'ELj32E' in r['symbol']]
    else:
        previous=load(w/'compiler_summary.json')
        targets=[r for r in previous['builds']['fetch_ids'] if r['whole_rows'] and 'ELj32E' in r['symbol']]
    proof=load(p/('compiler_runtime'+('_normal' if normal else '')+'_summary.json'))
    expected={(r['instruction_sha256'],r['instructions'],r['resource']) for r in targets}
    actual={(r['instruction_sha256'],r['instructions'],r['resource']) for r in proof['new_kernels']}
    assert actual==expected
    records.append(dict(build=proof['build'],unique_types=len(actual),
        matching_original_shared_id_target_instructions_and_resources=True,
        target_instruction_sha256=[r[0] for r in sorted(actual)],
        all_original_kernels_unchanged=proof['unchanged_original_kernels']))
result=dict(complete=True,records=records,
    candidate_header_sha256=hashlib.sha256((p/'runtime_ids.cuh').read_bytes()).hexdigest(),
    explanation='Only the cached target matches the firstfetch_ids machine instructions/resources. Dispatch retains cached-only routing and originaluncached kernels. This proves code identity, not fresh full-matrix performance or causation for predecessor negatives.',
    candidate_adopted=False,goal_complete=False)
(p/'predecessor_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
