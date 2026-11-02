import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'route_select_20261102'
assert not (p/'bench.py').exists()
for name in ['resources.py','plans.cpp','experiment.json','current.cuh','current_mla.cuh']:shutil.copy2(old/name,p/name)
s=(old/'bench.py').read_text()
needle='R, M, K, C, CAP = A.requests, A.mtp, META["topk"], 65536, META["capacity"]'
assert s.count(needle)==1;s=s.replace(needle,needle+'\nDISABLE_LRU=os.environ.get("TUNE_DISABLE_LRU")=="1"\nif DISABLE_LRU: CAP=0')
s=s.replace('if A.kind=="lru":w.set_host_cache','if A.kind=="lru" and CAP:w.set_host_cache').replace('if A.kind=="lru" and mode=="ours":w.release_host_cache','if A.kind=="lru" and CAP and mode=="ours":w.release_host_cache')
s=s.replace('configuration=dict(kind=A.kind,setting=setting','configuration=dict(kind="trace_nocache" if DISABLE_LRU else A.kind,cache_disabled=DISABLE_LRU,setting=setting')
(p/'bench.py').write_text(s)
(p/'.gitignore').write_text('__pycache__/\nplans.so\n');(p/'.gitattributes').write_text('*.log -whitespace\n')
(p/'manifest.json').write_text(json.dumps(dict(header_sha256=hashlib.sha256((p/'current.cuh').read_bytes()).hexdigest(),scope='Same full1024trace, originalfar fraction nearly1, MTP8R128K2048 BF16 h64 f24off; capacity0 vs prior current4096. Same tiers/model/timing/seed/bindings/reference. No CUDA changes.',benchmark_change='Set CAP=0; do not initialize/release cache; label trace_nocache. Existing offline planner supportscapacity0 and originalGraph/events/miss diagnostics remain.'),indent=2)+'\n')
