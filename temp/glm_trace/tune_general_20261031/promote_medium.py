"""Install the tested A100 dispatch predicate only after concrete experiment gates pass."""
raise SystemExit('Initial uniform 2-SM dispatch rejected: native low-traffic regression. Validate balanced_sm80 before creating a new promotion gate.')
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
g=load(p/'gpu3/audit.json');s=load(p/'audit_general.json')
assert g['gpu3_planned_performance_coverage_complete']
for name in ('gpu3_full_','gpu3_controls_','gpu3_nocache_','gpu3_confirmfull_'):assert g['coverage'][name]['complete']
affected=[r for r in g['pairs'] if r['cohort'] in ('gpu3_full_','gpu3_nocache_')]
assert len(affected)==220 and all(r['reduction']>0 for r in affected)
controls=[r for r in g['pairs'] if r['cohort']=='gpu3_controls_']
assert len(controls)==9 and min(r['reduction'] for r in controls)>-.01
new={r['setting']:r for r in s['rows'] if Path(r['raw']).name.startswith('medium_sxm_fullreverse_threshold2_')}
old={r['setting']:r for r in s['rows'] if Path(r['raw']).name.startswith('medium_sxm_fullreverse_base_')}
links={'g100_l0','g100_l20000','g50_l20000','nvlink','pcie'}
assert new.keys()==old.keys()==links
for k in links:
    assert new[k]['full_trace'] and old[k]['full_trace'] and new[k]['passes']==old[k]['passes']==5
    assert new[k]['mtp']==old[k]['mtp']==8 and new[k]['requests']==old[k]['requests']==32
    assert new[k]['fetch_ctas']==old[k]['fetch_ctas']==16 and not new[k]['host_split'] and not old[k]['host_split']
    assert new[k]['ours_us']<old[k]['ours_us']
retained=p/'threshold2_sm80.cuh';test=load(p/'tests_threshold2_sm80_gpu3.json')
assert test['complete'] and test['exit_code']==test['compile_exit_code']==0
assert test['header_sha256']==sha(retained)
assert test['test_source_sha256']==sha(root/'tests/attention/test_mla_host_batch.py')
assert test['log_sha256']==sha(p/'pytest_threshold2_sm80_gpu3.log')
assert '105 passed' in (p/'pytest_threshold2_sm80_gpu3.log').read_text()
assert not test['resources_start']['foreign'] and not test['resources_end']['foreign']
compiler=load(p/'compiler_summary.json')
for name in ('threshold2','threshold2_sm80'):
    assert all(r['instruction_text_equal'] and r['resource_equal'] for group in compiler['comparisons'][name].values() for r in group)
normal=load(p/'compiler_normal_summary.json')
assert normal['all_18_kernel_instruction_text_and_resources_equal'] and normal['retained_header_sha256']==sha(retained)
base=(p/'base.cuh').read_text()
assert retained.read_text()==base.replace('params.num_rows >= 4u * num_sms &&','params.num_rows >= (CTA_TILE_KV == 32 ? 2u : 4u) * num_sms &&')
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
header=root/'include/flashinfer/attention/mla_host_batch.cuh'
assert header.read_bytes() in ((p/'base.cuh').read_bytes(),retained.read_bytes()),'Do not overwrite unknown dirty source'
record=dict(timestamp=time.strftime('%FT%T%z'),goal_complete=False,scope='A100 CTA_TILE_KV=32 staged/split-off/64-head/full-staging dispatch eligibility changes from 4 to 2 SM waves. Other shared-memory configurations retain the original threshold.',
    kind='dispatch improvement; same GPU math/LRU code and same caller parameters',
    base_header_sha256=sha(p/'base.cuh'),retained_header_sha256=sha(retained),
    gpu3_audit_sha256=sha(p/'gpu3/audit.json'),sxm_audit_sha256=sha(p/'audit_general.json'),
    tests_sha256=sha(p/'tests_threshold2_sm80_gpu3.json'),emulated_compiler_sha256=sha(p/'compiler_summary.json'),normal_compiler_sha256=sha(p/'compiler_normal_summary.json'),
    affected_paired_cells=220,min_affected_reduction=min(r['reduction'] for r in affected),
    unaffected_control_max_slowdown=max(-r['reduction'] for r in controls),
    sxm_five_pass={k:dict(old_us=old[k]['ours_us'],new_us=new[k]['ours_us'],reduction=1-new[k]['ours_us']/old[k]['ours_us'],memcpy_us=new[k]['memcpy_us'],local_us=new[k]['local_us'],reference_us=new[k]['reference_us'],over_reference=new[k]['over_reference']) for k in sorted(links)},
    remaining='Large-batch 100 GB/s 32–34% gap is unchanged. No H800 performance evidence, no full-goal completion claim.',adopted=False)
(p/'promotion_gate.json').write_text(json.dumps(record,indent=2)+'\n')
header.write_bytes(retained.read_bytes())
record['adopted']=True;record['production_header_sha256']=sha(header)
(p/'promotion.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
