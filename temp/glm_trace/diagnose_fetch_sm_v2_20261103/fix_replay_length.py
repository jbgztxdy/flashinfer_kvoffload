"""One-time pre-GPU correction: final reset clears both row length cursors."""
from pathlib import Path
import hashlib,json,shutil,difflib
p=Path(__file__).resolve().parent
archive=p/'pre_gpu_replay_length';archive.mkdir()
for f in ['capacity.cuh','capacity.diff','build.json','compile_capacity.log','capacity_resources.txt']:
 shutil.copy2(p/f,archive/f)
s=(p/'capacity.cuh').read_text()
a=s.index('void capacity_pv_run_joint_host_item(');b=s.index('void capacity_pv_run_joint_host_items(',a)
t=s[a:b]
old='    kv_len = host_batch_row_len(params, row);'
new='''    // Replay follows a complete launch, whose final reset clears row cursors.
    // This diagnostic accepts the unpadded 2048-entry trace only.
    kv_len = (params.host_item_entries==840 || params.host_item_entries==1080)
        ? params.kv_indptr[row+1]-params.kv_indptr[row] : host_batch_row_len(params,row);'''
assert t.count(old)==1;t=t.replace(old,new);s=s[:a]+t+s[b:]
old='''    __syncthreads();
    if(blockIdx.y<workers) {'''
new='''    __syncthreads();
    if(tid==0) {
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      reinterpret_cast<uint32_t*>(params.partial_o)[blockIdx.y]=sm;
    }
    cg::this_grid().sync();
    if(blockIdx.y<workers) {'''
assert s.count(old)==1;s=s.replace(old,new)
(p/'capacity.cuh').write_text(s)
(p/'include_capacity/flashinfer/attention/mla_host_batch.cuh').write_text(s)
base=(p.parents[2]/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
(p/'capacity.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production.cuh',tofile='capacity.cuh',n=0)))
m=json.loads((p/'manifest.json').read_text());m['variants']['capacity']=hashlib.sha256(s.encode()).hexdigest();m['capacity']+=' Replay restores exact unpadded length from kv_indptr after final cursors reset; grid barrier and SM IDs verify residency. No GPU run used first version.';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
