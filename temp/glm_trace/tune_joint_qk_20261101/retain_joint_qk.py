"""Install only the frozen, equivalent source after all planned regression/confirmation gates."""
import argparse,hashlib,json,shutil,statistics,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');args=parser.parse_args()
base='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
measured='a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
math='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
matrix='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(p/'base.cuh')==base and sha(p/'joint_qk.cuh')==measured
candidate=sha(p/'retained_joint_qk.cuh');manifest=read(p/'retained_source_manifest.json')
assert manifest['measured_header_sha256']==measured and manifest['reviewable_header_sha256']==candidate
assert sha(root/'include/flashinfer/attention/mla.cuh')==sha(p/'retained_joint_qk_mla.cuh')==math
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')==matrix
coverage=read(p/'coverage.json');audit=read(p/'performance_audit.json')
assert coverage['audit_sha256']==sha(p/'performance_audit.json')
assert coverage['gpu3_matrix_complete'] and coverage['sxm_matrix_complete']
assert len(coverage['groups'])==4 and sum(g['expected_cells'] for g in coverage['groups'])==156
assert not audit['pending_references'] and not audit['incomplete']
index={(r['cohort'],r['setting'],r['variant']):r for r in audit['rows']}
confirm=read(p/'sxm_confirmation_plan.json');assert len(confirm['stages'])==3
confirmations=[]
for stage in confirm['stages']:
 for setting in stage['settings']:
  values=[index[stage['tag'],setting,v] for v in ['base','joint_mdo_typed','joint_qk']]
  assert all(r['hardware']=='SXM0/1' and r['positions']==1024 and r['passes']==5 and r['fetch_ctas']==24 and r['mtp']==8 and r['requests']==128 for r in values)
  b,j,n=[r['ours_us'] for r in values];reduction=1-n/b;vsjoint=1-n/j
  if setting.startswith('g100'):assert reduction>.01, (stage['tag'],setting,reduction)
  else:assert reduction>=-.01,(stage['tag'],setting,reduction)
  confirmations.append(dict(cohort=stage['tag'],setting=setting,base_us=b,joint_mdo_us=j,joint_qk_us=n,reduction=reduction,reduction_vs_joint_mdo=vsjoint))
for cohort in ['compositionfull1024_f24_']:
 for setting in ['g100_l0','g100_l20000']:
  b,j,n=[index[cohort,setting,v]['ours_us'] for v in ['base','joint_mdo_typed','joint_qk']]
  assert n<b and n<j
controls=read(p/'controls_audit.json');assert controls['complete'] and controls['all_within_one_percent_each_endpoint']
assert controls['performance_audit_sha256']==sha(p/'performance_audit.json')
allowed={("joint_qk_shapes_gpu3_native_",1,256,'lru','native_hbm',None),("joint_qk_shapes_gpu3_native_",8,64,'nocache','native_hbm',0.)}
slow=[r for r in audit['pairs'] if r['variant']=='joint_qk' and r['cohort'].startswith('joint_qk_shapes_') and r['reduction']<-.01]
assert {(r['cohort'],r['mtp'],r['requests'],r['kind'],r['setting'],r['fraction']) for r in slow}<=allowed,slow
changed=[r for r in audit['pairs'] if r['variant']=='joint_qk' and r['cohort'].startswith('joint_qk_shapes_') and r['kind']=='lru' and r['mtp']*r['requests']>=432]
assert len(changed)==60 and statistics.mean(r['reduction'] for r in changed)>0
assert all(r['reduction']>=-.01 for r in changed)
for suffix,n in [('',34),('_normal',68)]:
 proof=read(p/('compiler_joint_qk'+suffix+'_summary.json'));assert proof['complete'] and proof['unchanged_original_kernels']==n and proof['candidate_header_sha256']==measured and proof['math_header_sha256']==math
 bridge=read(p/('retained_codegen'+suffix+'_summary.json'));assert bridge['complete'] and bridge['all_original_and_new_targets_identical'] and bridge['unchanged_kernel_instances']==n+(2 if not suffix else 4)
wordproof=read(p/'machine_word_identity.json');assert wordproof['complete'] and len(wordproof['builds'])==2
assert all(x['original_all_words_identical'] and x['cleanup_all_words_identical'] for x in wordproof['builds'])
for variant,expected in [('joint_qk',measured),('retained_joint_qk',candidate)]:
 tests=read(p/('tests_'+variant+'_gpu3.json'));assert tests['complete'] and tests['normal_build'] and tests['exit_code']==tests['compile_exit_code']==0 and tests['header_sha256']==expected
 assert tests['log_sha256']==sha(p/('pytest_'+variant+'_gpu3.log')) and '105 passed' in (p/('pytest_'+variant+'_gpu3.log')).read_text()
 stress=read(p/('stress_'+variant+'_gpu3.json'));assert stress['complete'] and stress['exit_code']==0 and stress['tests']==4 and stress['header_sha256']==expected
 assert stress['log_sha256']==sha(p/('pytest_stress_'+variant+'.log')) and '4 passed' in (p/('pytest_stress_'+variant+'.log')).read_text()
for filename,physical in [('dispatch_probe_joint_qk.json',3),('dispatch_probe_joint_qk_sxm.json',0)]:
 probe=read(p/filename);assert probe['header_sha256']==measured and probe['physical_gpu']==physical and probe['graph_nodes']==probe['kernel_nodes']==1
 cfg=probe['launches'][0];assert cfg['grid']==[1,108,1] and cfg['block']==[32,4,2] and cfg['active_blocks_per_SM']==1 and cfg['dynamic_shared_bytes']==147968
production=root/'include/flashinfer/attention/mla_host_batch.cuh';assert sha(production) in [base,candidate],'Unrelatedproductionedit'
files=['performance_audit.json','coverage.json','controls_audit.json','controls_plan.json','manifest.json','retained_source_manifest.json','machine_word_identity.json','sxm_confirmation_plan.json','composition_comparisons.json','dispatch_probe_joint_qk.json','dispatch_probe_joint_qk_sxm.json','compiler_joint_qk_summary.json','compiler_joint_qk_normal_summary.json','retained_codegen_summary.json','retained_codegen_normal_summary.json','tests_joint_qk_gpu3.json','tests_retained_joint_qk_gpu3.json','stress_joint_qk_gpu3.json','stress_retained_joint_qk_gpu3.json','retained_joint_qk.cuh','joint_qk.cuh']
snapshot=p/'retention_snapshot';snapshot.mkdir(exist_ok=True);frozen=[]
for name in files:
 f=p/name;target=snapshot/(sha(f)+'_'+name)
 if target.exists():assert target.read_bytes()==f.read_bytes()
 else:shutil.copy2(f,target)
 frozen.append(dict(path=str(f.relative_to(root)),sha256=sha(f),snapshot=str(target.relative_to(root))))
record=dict(timestamp=time.strftime('%FT%T%z'),candidate_adopted=bool(args.install),goal_complete=False,
 base_header_sha256=base,measured_header_sha256=measured,candidate_header_sha256=candidate,
 matrix_cells=156,matrix_pairs=78,changed_lru_pairs=60,confirmations=confirmations,
 changed_matrix_minimum_reduction=min(r['reduction'] for r in changed),changed_matrix_mean_reduction=statistics.mean(r['reduction'] for r in changed),changed_matrix_maximum_reduction=max(r['reduction'] for r in changed),
 slow_original_cells=slow,control_brackets=controls['cases'],frozen=frozen,
 slow_cell_policy='Alloriginalnegativerawretained. Two>1%unchangedGPU3pathsfailedtoreproducein5passbrackets,eachwithin1%ofbothbaselines;alloldmachinewordsincludingcontrolwordsexact. No promiseeveryindependentallocationfaster. Anyother>1%matrixnegativeblocksthisinstaller.',
 limitation='SinglecooperativekernelandexactLRU/payloadremain,attentionequationunchangedbutFP32onlinegroupingdiffers. H800unvalidatedunchangeddispatch. Current100GBgapstillabove10%target;goalactive. Freshrootnormalcompile/testsrequiredafterinstall.')
if args.install:production.write_bytes((p/'retained_joint_qk.cuh').read_bytes());assert sha(production)==candidate
(p/'joint_qk_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print({k:record[k] for k in ['candidate_adopted','goal_complete','candidate_header_sha256','matrix_cells','matrix_pairs','changed_lru_pairs','changed_matrix_minimum_reduction','changed_matrix_mean_reduction','changed_matrix_maximum_reduction']})
