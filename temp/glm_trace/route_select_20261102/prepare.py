import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_qk_cursor_20261102';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not (p/'current.cuh').exists()
for dest,src in [('current.cuh','qk_cursor.cuh'),('current_mla.cuh','qk_cursor_mla.cuh')]:shutil.copy2(old/src,p/dest)
for name in ['resources.py','plans.cpp','experiment.json']:
 shutil.copy2(p.parent/'tune_stream_source_20261101'/name,p/name)
s=(p.parent/'tune_stream_source_20261101/bench.py').read_text()
s=s.replace('VARIANT_INCLUDE=DEST/("include_"+VARIANT)','VARIANT_INCLUDE=DEST.parent/"tune_qk_cursor_20261102/include_qk_cursor"')
assert 'if A.route=="staged" else' in s
(p/'bench.py').write_text(s)
libs=[]
for mode in ['emu','normal']:
 ws=old/('jit_qk_cursor'+('_normal' if mode=='normal' else ''))
 for f in ws.rglob('*.so'):libs.append(dict(mode=mode,path=str(f.relative_to(root)),sha256=sha(f)))
assert len(libs)==4
(p/'manifest.json').write_text(json.dumps(dict(header_sha256=sha(p/'current.cuh'),math_sha256=sha(p/'current_mla.cuh'),reused_libraries=libs,mechanism='No CUDA modification; matched current staged/inplace route comparison. Explicit route per shape/tier/cache regime.',scope='SXM0/1 NUMA0; shared gpu01 lock; Graph/event timing and exact LRU maintained.'),indent=2)+'\n')
(p/'.gitignore').write_text('__pycache__/\nplans.so\n')
(p/'.gitattributes').write_text('*.log -whitespace\n')
print('PREPARED_CURRENT_SOURCE_AND_BINARIES')
