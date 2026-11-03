"""Read-only final evidence audit and migration checkpoint; no GPU or compiler starts."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda name:json.loads((p/name).read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
a=load('performance_audit.json');m=load('smem_manifest.json')
assert len(a['pairs'])==6 and not a['incomplete']
assert all(x['reduction']<0 for x in a['pairs'])
decisions=dict(timestamp=time.strftime('%FT%T%z'),variants={})
for v,mf,tests in [('square','manifest.json',2),('square_p','direct_manifest.json',2)]:
 manifest=load(mf);pairs=[x for x in a['pairs'] if x['variant']==v];assert len(pairs)==3
 decisions['variants'][v]=dict(decision='rejected',adopted=False,header_sha256=manifest['candidate_sha256'],targeted_tests=tests,pairs=pairs,reason='All3points slower than same-cohort frozenproduction. Do not expand or rerun.')
(p/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
record=dict(timestamp=time.strftime('%FT%T%z'),status='handoff_user_requested_no_active_runner',
 production_sha256=sha(root/'include/flashinfer/attention/mla_host_batch.cuh'),
 math_sha256=sha(root/'include/flashinfer/attention/mla.cuh'),far_emu_sha256=sha(root/'include/flashinfer/attention/far_link_emu.cuh'),
 current_candidate='square_smem',current_candidate_sha256=m['candidate_sha256'],candidate_installed=False,
 candidate_state='Original30e7draft compilation failed: new bool square_smem collides with old control constexpr square_smem. Renamed bool square_smem_enabled in frozenheader/include/generator; finalc69a2643 is UNBUILT and has noGPUtests/timings. Oldfailed draft/logs/buildrecord infailed_smem_dispatch/.',
 previous_measured_states='shared_claim rejected freshGPU3; anchor rejected; square/square_p rejected; see siblingdecisions and currentperformance_audit.',
 sourcefiles=['square.inc','direct_p.inc','make_smem_pe.py','square_smem.cuh','smem_manifest.json','smem_pe.diff'],
 terminal_exec_handles={'shared_claim_gpu3':65933,'anchor_initial_build':9696,'anchor_failed_validation':78482,'anchor_fixed_build':8965,'anchor_final_screen':92066,'square_build':50528,'square_screen':10809,'square_p_failed_build':30097,'square_p_fixed_build':63040,'square_p_screen':48075,'square_smem_failed_build':34586},
 live_task_runners=[],signals_sent=[],
 next_step='FreshGPU3preflight+sharedlock; runPython build_smem.py only. Ifsuccess inspectactualREG/STACKandnewtypes, then run_smem.py for2focusedtests+3freshsamehardwarepairs. run_smem.py doesnotrepeatoldercontrolcohorts. Then audit_performance.py. Do notrerunmake_smem_pe.py orfixscripts thatassertnewpaths.',
 limits='NoGPUproofafterfix,noFP16/SXM/NVLink/H800newtiming,newGraphnodeprobe/sanitizer orfullbinaryidentityaudit. Do notextrapolate.',
 h800_query='h800_login_query_20261103.txt',goal_complete=False)
assert record['production_sha256']=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert record['current_candidate_sha256']==sha(p/'square_smem.cuh')==sha(p/'include_square_smem/flashinfer/attention/mla_host_batch.cuh')
(p/'migration_state.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
