"""Install only the measured header after the targeted evidence is complete."""
import hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest=json.loads((p/'manifest.json').read_text());audit=json.loads((p/'performance_audit.json').read_text());probe=json.loads((p/'dispatch_probe_pv_feature.json').read_text())
assert json.loads((p/'state.json').read_text())['complete']
assert json.loads((p/'confirmation_state.json').read_text())['complete']
assert '2 passed' in (p/'targeted_correctness.log').read_text()
assert len(audit['pairs'])==6 and not audit['incomplete']
assert all(x['reduction']>0 for x in audit['pairs'] if x['setting'].startswith('g100') or x['setting']=='native_hbm')
assert len([x for x in audit['pairs'] if x['cohort']=='reverse256_'])==2
assert probe['kernel_nodes']==1 and probe['header_sha256']==manifest['candidate_sha256']
assert probe['launches'][0]['grid']==[1,108,1] and 'BatchMLAPVFeatureKernel' in probe['launches'][0]['kernel_name']
assert sha(p/'pv_feature.cuh')==manifest['candidate_sha256']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==manifest['base_sha256']
assert sha(root/'include/flashinfer/attention/mla.cuh')==manifest['math_sha256']
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
manifest['adopted']=True;(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
r=dict(timestamp=time.strftime('%FT%T%z'),candidate_adopted=True,goal_complete=False,base_sha256=manifest['base_sha256'],production_sha256=manifest['candidate_sha256'],exact_measured_bytes=True,pairs=audit['pairs'],evidence=[],validation_scope='2targetedBF16tests+actualsinglekernelGraph+6pairedshortpoints. Reuse unchanged math/fetch/LRU source; no fullsuite, sanitizer, fullwindow, FP16/SXM/H800/nativePCIe/NVLink or fulloriginalbinaryidentity claims.',policy='User-authorized lightweight iteration: verified mechanism and repeated100GBgains permit retention; broader coverage remains open.')
for name in ['manifest.json','pv_feature.cuh','performance_audit.json','source_review.json','codegen_review.json','schedule_review.json','targeted_correctness.log','state.json','confirmation_state.json','dispatch_probe_pv_feature.json']:
 f=p/name;digest=sha(f);target=p/'retention_snapshot'/(digest+'_'+name);target.parent.mkdir(exist_ok=True);shutil.copy2(f,target)
 r['evidence'].append(dict(file=name,sha256=digest,snapshot=str(target.relative_to(p))))
(root/'include/flashinfer/attention/mla_host_batch.cuh').write_bytes((p/'pv_feature.cuh').read_bytes())
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==r['production_sha256']
(p/'retention.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(production_sha256=r['production_sha256'],candidate_adopted=True,goal_complete=False),indent=2))
