"""Precompute final KV row pointers in unused whole-row partial storage, inside existing phases."""
import hashlib,json,re,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
control=(p.parent/'tune_joint_qk_20261101/joint_qk.cuh').read_text()
assert sha(base)=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(control)=='a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
def function(s,name):
 m=re.search(r'\b(?:void|uint32_t) '+re.escape(name)+r'\(',s);assert m,name
 a=s.rfind('template <',0,m.start());i=s.index('{',m.end());d=1;j=i+1
 while d:d+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
helper='''// Whole rows never consume partial states. The launcher bounds this uint64 table by the
// actual partial allocation; the original barriers publish its writes with the original remap.
template <typename Params>
__device__ __forceinline__ uint64_t* addressed_row_table(const Params& params) {
  return reinterpret_cast<uint64_t*>(params.partial_o);
}
template <typename KTraits, typename Params>
__device__ __forceinline__ uint64_t addressed_page_pointer(
    const Params& params, typename Params::IdType page) {
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  const bool staged = page < 0;
  const int64_t row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
  return reinterpret_cast<uint64_t>((staged ? params.staging : params.ckv) + row * ROW);
}

'''
prepare=function(base,'prepare_host_batch_rows').replace('prepare_host_batch_rows(', 'prepare_addressed_rows(',1)
key='          params.remap[begin + length - 1 - local_first - __popc(local_masks[i] & lower)] = pages[i];'
assert prepare.count(key)==1
prepare=prepare.replace(key,key+'''
          addressed_row_table(params)[begin + length - 1 - local_first - __popc(local_masks[i] & lower)] =
              addressed_page_pointer<KTraits>(params, pages[i]);''')
resolve=function(base,'resolve_host_batch_slots').replace('resolve_host_batch_slots(', 'resolve_addressed_slots(',1)
key='        params.remap[begin + j] = host_batch_staged_entry(params, slot);'
assert resolve.count(key)==1
resolve=resolve.replace(key,key+'''
        addressed_row_table(params)[begin + j] =
            addressed_page_pointer<KTraits>(params, host_batch_staged_entry(params, slot));''')
compact=function(base,'compact_host_batch_rows').replace('compact_host_batch_rows(', 'compact_addressed_rows(',1)
key='      params.remap[begin + host + k] = params.remap[begin + movers + k];'
assert compact.count(key)==1
compact=compact.replace(key,key+'''
      addressed_row_table(params)[begin + host + k] = addressed_row_table(params)[begin + movers + k];''')
read=function(base,'read_host_batch_pages').replace('read_host_batch_pages(', 'read_addressed_pages(',1)
read=read.replace('typename KTraits::IdType* pages','uint64_t* pages')
key='    pages[j] = kv_idx < kv_len ? params.remap[index_base + kv_idx] : 0;'
assert read.count(key)==1
read=read.replace(key,'''    pages[j] = kv_idx < kv_len ? addressed_row_table(params)[index_base + kv_idx]
                               : reinterpret_cast<uint64_t>(params.ckv);''')
load=function(base,'load_host_batch_kv').replace('load_host_batch_kv(', 'load_addressed_kv(',1)
load=load.replace('const typename KTraits::IdType* pages','const uint64_t* pages')
key='    resolve_host_batch_row<KTraits>(params, pages[j], ckv_ptr, kpe_ptr);'
assert load.count(key)==1
load=load.replace(key,'''    ckv_ptr = reinterpret_cast<DTypeKV*>(pages[j]) +
              (lane_idx % 8) * upcast_size<DTypeKV>();
    kpe_ptr = ckv_ptr + KTraits::HEAD_DIM_CKV;''')
# This helper is only instantiated for fully staged AddressedHostParams, so the original
# writeback/FAR direct-load branches are compile-time inactive, as in the control.
s=control.replace('JointQKTraits','AddressedQKTraits').replace('JointQKHostParams','AddressedHostParams').replace('BatchMLAJointQKKernel','BatchMLAAddressedKernel')
key='// Two different KV tiles share each Q fragment.'
assert s.count(key)==1
s=s.replace(key,helper+'\n\n'.join([prepare,resolve,compact,read,load])+'\n\n'+key,1)
old=function(s,'attend_joint_host_tiles');new=old.replace('typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];','uint64_t pages[kHostBatchTileRows<KTraits>];').replace('read_host_batch_pages<KTraits>','read_addressed_pages<KTraits>').replace('load_host_batch_kv<KTraits>','load_addressed_kv<KTraits>')
assert old!=new;s=s.replace(old,new,1)
old=function(s,'BatchMLAAddressedKernel');new=old.replace('prepare_host_batch_rows<KTraits>','prepare_addressed_rows<KTraits>').replace('resolve_host_batch_slots<KTraits>','resolve_addressed_slots<KTraits>').replace('compact_host_batch_rows<KTraits>','compact_addressed_rows<KTraits>')
assert old!=new;s=s.replace(old,new,1)
key='''        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;'''
assert s.count(key)==1
s=s.replace(key,'''        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len &&
        // The original planner reserves partial_capacity states of 64 heads x 512 features.
        static_cast<uint64_t>(params.num_rows) * max_kv_len * sizeof(uint64_t) <=
            static_cast<uint64_t>(partial_capacity) * 64u * 512u * sizeof(DTypeO);''',1)
for v,host in [('base',base),('joint_qk',control),('address_map',s)]:
 for filename,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
  f=p/filename
  if f.exists():assert f.read_text()==value
  else:f.write_text(value)
 for suffix in ['', '_normal']:
  d=p/('include_'+v+suffix)
  if not d.exists():shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
  assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(host)
  assert sha((d/'flashinfer/attention/mla.cuh').read_text())==sha(math)
m=dict(base_sha256=sha(base),joint_qk_control_sha256=sha(control),candidate_sha256=sha(s),math_sha256=sha(math),
 mechanism='Precompute finalnear/cachedpointers withoriginalpreparestores, finalstagedpointers withoriginalresolveremapstores, copy pointeralongwithdisjointpaddingmovers. Existinggridbarrierspublishbothremapandtable; no newkernel/barrier/allocation/API. Tile prefetch reads uint64pointer and loads direct row+featureoffset, originalreadyflag acquire stillreadsint32remap. SamejointQK/MDO/PV/fetch/LRU.',
 workspace_guard='Onlyfullystagedcachedwholerow >=4SM; uint64(rows)*max_kv_len*8 <= uint64(partial_capacity)*64*512*sizeof(DTypeO). Whole-row outputneverusespartial_o; insufficientcapacityretainsoriginal productionpath. Paddingbeyondkv_len usesvaliddummybaseandzero-fillpredicate.',
 risks='Extra8B/NNZwrites and8Btile-IDreads versus4B, pointerregisterpressure; noassumedperformancegain. Originalprepare/resolve/compact/mathhelpersliteral, dedicatedtrait/params isolation. Requireold34/68exactproof,actualnewGraph,normal105+4testsplusworkspacefallbackguard beforetiming.',
 distinction='Oldprebind bindsfetchcachemetadata; direct_cache changesattentiondatasource. This onlyprecomputes existingrowaddresses inalreadyallocatedunusedpartialstorage, staging/cachelayoutandpayloadunchanged.',candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
