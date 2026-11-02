"""Install the exact measured cursor header using focused evidence, without broad gate repetition."""
import hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
load=lambda f:json.loads(f.read_text())
m=load(p/'manifest.json');a=load(p/'performance_audit.json');g=load(p/'dispatch_probe_stream_source.json');c=load(p/'codegen.json')
assert len(a['pairs'])==8 and len(a['rows'])==16 and not a['incomplete'] and not a['pending_references']
assert all(x['reduction']>=0 for x in a['pairs'])
assert all(x['reduction']>.01 for x in a['pairs'] if x['cohort']=='reverse1024_' and x['setting'].startswith('g100'))
for f in ['focused_state.json','confirmation_state.json']:
 d=load(p/f);assert d['complete'] and all(x['exit_code']==0 for x in d['actions'])
assert '4 passed, 6 deselected' in (p/'targeted_correctness.log').read_text()
for mode in ['emu','normal']:
 assert c[mode]['unchanged']==38 and len(c[mode]['changed'])==2
 for variant,libs in c[mode]['libraries'].items():
  assert all(sha(p/x['path'])==x['sha256'] for x in libs)
assert g['kernel_nodes']==1 and g['launches'][0]['grid']==[1,108,1] and g['launches'][0]['block']==[32,4,2]
assert g['header_sha256']==m['candidate_sha256']==sha(p/'qk_cursor.cuh')
assert sha(root/'include/flashinfer/attention/mla.cuh')==m['math_sha256']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
target=root/'include/flashinfer/attention/mla_host_batch.cuh';assert sha(target)==m['base_sha256']
snapshot=p/'retention_snapshot';snapshot.mkdir(exist_ok=True);evidence=[]
for name in ['manifest.json','performance_audit.json','codegen.json','targeted_correctness.log','focused_state.json','confirmation_state.json','dispatch_probe_stream_source.json','qk_cursor.cuh']:
 f=p/name;h=sha(f);dest=snapshot/(h+'_'+name);shutil.copy2(f,dest);evidence.append(dict(file=name,sha256=h,snapshot=str(dest.relative_to(p))))
target.write_bytes((p/'qk_cursor.cuh').read_bytes());assert sha(target)==m['candidate_sha256']
r=dict(timestamp=time.strftime('%FT%T%z'),candidate_adopted=True,goal_complete=False,base_sha256=m['base_sha256'],production_sha256=sha(target),exact_measured_bytes=True,evidence=evidence,pairs=a['pairs'],validation_scope='4 BF16 targeted payload/LSE/Graph tests; integer address oracle16384checks; original38BF16kernel word/resource identities per EMU/normal build; actualsingle108CTA256threadGraph; short3point + reversefull1024x3 + normalHBM/PCIe256x3.',not_covered='No newFP16/SXM/NVLink/H800, broadmatrix, sanitizerrepeat orfreshrootbuild/test.',risk='Small gains on oneGPU and long-lived address state increases stack to16B in target. Large100GB remains above reference; no all-scenario guarantee.')
(p/'retention.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(production_sha256=r['production_sha256'],candidate_adopted=True,pairs=len(a['pairs']),goal_complete=False),indent=2))
