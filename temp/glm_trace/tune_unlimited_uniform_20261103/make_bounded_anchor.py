from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'rescale_skip.cuh').read_text()
start=s.index('__device__ __forceinline__ void clean_service_update_joint_host_mdo(');head=s[:start];tail=s[start:]
a='''    const float factor = math::ptx_exp2(previous[j]*scale - m[j]*scale);'''
b='''    // m is a valid softmax normalization origin, not necessarily the maximum.
    // Hold it until current log2 scores exceed it by8. Thus P <=256, well
    // within BF16 range, and O/d only rescale on an actual anchor transition.
    if ((m[j] - previous[j])*scale <= 8.f) m[j] = previous[j];
    const float factor = math::ptx_exp2(previous[j]*scale - m[j]*scale);'''
assert tail.count(a)==1;s=head+tail.replace(a,b)
assert not(p/'bounded_anchor.cuh').exists();(p/'bounded_anchor.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_bounded_anchor');(p/'include_bounded_anchor/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_bounded_anchor.py').write_text((p/f'{n}_rescale_skip.py').read_text().replace('rescale_skip','bounded_anchor'))
f=p/'test_bounded_anchor.py';t=f.read_text().replace("'length,rows,staged',[(65,512,True),(2048,512,True)]","'length,rows,staged,qscale',[(65,512,True,1),(2048,512,True,1),(2048,512,True,.05),(2048,512,True,4),(2048,512,True,32)]").replace('(length,rows,staged):','(length,rows,staged,qscale):').replace('    idx=torch.empty(', '    q.mul_(qscale);qp.mul_(qscale)\n    idx=torch.empty(')
f.write_text(t)
(p/'bounded_anchor_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'rescale_skip.cuh').read_bytes()).hexdigest(),mechanism='bounded softmax normalization anchor: update only if current max exceeds previous anchor by8log2units; BF16 P<=256; true QK/PV and exactLRU/service unchanged; same tolerance'),indent=2)+'\n')
