"""Verify both service implementations and the unchanged production/controls."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=read(p/'manifest.json');sm=read(p/'shared_manifest.json')
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
assert sha(root/'include/flashinfer/attention/mla.cuh')==m['math_sha256']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
headers=dict(base=m['base_sha256'],control_qp=m['control_sha256'],qks=m['candidate_sha256'],qks_shared=sm['candidate_sha256'])
for v,h in headers.items():
 assert sha(p/(v+'.cuh'))==h and sha(p/(v+'_mla.cuh'))==m['math_sha256']
 d=p/('include_'+v)/'flashinfer/attention'
 if d.exists():assert sha(d/'mla_host_batch.cuh')==h and sha(d/'mla.cuh')==m['math_sha256']
control=read(p/'reused_qp_control.json');assert control['complete'] and control['additional_tests_counted']==0
for item in control['proofs']:assert sha(root/item['path'])==item['sha256']
proofs=[];probes=[]
for v,mem in [('qks',158464),('qks_shared',164224)]:
 for suffix,count in [('',40),('_normal',80)]:
  f=p/('compiler_'+v+suffix+'_summary.json');d=read(f)
  assert d['complete'] and d['unchanged_original_kernels']==count and d['all_original_instruction_and_control_words_identical']
  assert d['base_header_sha256']==headers['control_qp'] and d['candidate_header_sha256']==headers[v]
  for variant,libs in d['libraries'].items():
   directory=p.parent/'tune_qk_pv_roles_20261101'/('jit_qkpv_roles'+suffix) if variant=='base' else p/('jit_'+variant+suffix)
   for lib in libs:
    files=list(directory.rglob(lib['name']));assert len(files)==1 and sha(files[0])==lib['sha256']
  proofs.append(dict(file=f.name,sha256=sha(f),original_kernels=count,resources=[x['resource'] for x in d['new_kernels']]))
 f=p/('dispatch_probe_'+v+'.json');d=read(f);launch=d['launches'][0]
 assert d['header_sha256']==headers[v] and d['kernel_nodes']==1
 assert launch['grid']==[1,108,1] and launch['block']==[32,4,3] and launch['dynamic_shared_bytes']==mem and launch['active_blocks_per_SM']==1
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 for kind in ['synccheck','racecheck','memcheck']:
  text=(p/(kind+'_'+v+'.log')).read_text()
  assert 'ERROR SUMMARY: 0 errors' in text or (kind=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text)
 probes.append(dict(file=f.name,sha256=sha(f),launch=launch,captures=len(list((p/('dispatch_probes_'+v)).glob('*.json')))))
for name in ['validation_state.json','validation_shared_state.json']:
 d=read(p/name);assert d['complete'] and d['phase']=='short_screen_complete' and all(x['exit_code']==0 for x in d['steps'])
a=read(p/'performance_audit.json');initial=read(p/'initial_performance_audit.json')
assert len(a['rows'])==21 and len(a['pairs'])==15 and not a['incomplete'] and not a['pending_references']
assert a['timed_launches']==4032 and a['accepted_segments']==252
assert sum(x['tests'] for x in a['tests']+a['stress_tests'])==222
assert len(initial['rows'])==9 and len(initial['pairs'])==6
for row in initial['rows']:
 match=[r for r in a['rows'] if r['raw']==row['raw']];assert match==[row]
for item in initial['files']:assert sha(root/item['path'])==item['sha256']
pre=read(p/'precompile_review.json');assert sha(p/'precompile_review/qks.cuh')==pre['old_header_sha256']
layout=read(p/'shared_job_review.json');assert layout['complete'] and layout['candidate_sha256']==headers['qks_shared']
record=dict(complete=True,timestamp=time.strftime('%FT%T%z'),goal_complete=False,candidate_adopted=False,production_unchanged=True,
 headers=headers,proofs=proofs,probes=probes,rows=21,pairs=15,timed_launches=4032,accepted_segments=252,numerical_tests=222,
 reused_control_tests_counted=0,transitive_identity='Eachcandidate preserves QP40/80 literally; verified reusedQPproof preserves CPE38/76 literally.',
 limitations='Short256position/3coldpass GPU3 screen; no fulltrace/nativeNVLink/SXM/adoption proof. Sanitizers firstmatched BF16/K97 only; memcheck API reporting disabled,device checks retained.',
 parent_cached_pe_phase=read(p.parent/'tune_cached_pe_20261101/broader_validation_state.json')['phase'])
(p/'checkpoint_audit.json').write_text(json.dumps(record,indent=2)+'\n');print('CHECKPOINT_VERIFIED',record['rows'],record['numerical_tests'])
