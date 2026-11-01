"""Add incremental far-service to the four QK-role warps, preserving all prior code."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];prior=p.parent/'tune_qk_pv_roles_20261101';cpe=p.parent/'tune_cached_pe_20261101'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
control=(prior/'qkpv_roles.cuh').read_text();base=(cpe/'cached_pe.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(control)=='1252cdee93be85dd74f9fdd9a922b28e5327c3929dfbece1811021bf015648f4'
assert sha(base)=='a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
def function(source,name):
 pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos);first=source.index('{',pos);depth=1;end=first+1
 while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[start:end]+'\n\n'
def rename(s):return s.replace('qkpv_','qks_').replace('QKPV','QKS')
helpers=rename((prior/'qkpv.inc').read_text())
helpers=helpers.replace('      alignas(16) float o_scale[64], m_final[64], d_final[64];','      alignas(16) float o_scale[64], m_final[64], d_final[64];\n      alignas(128) uint4 service_payload[2][576];')
helpers=helpers.replace('    // Fetch must retain the original two64-row buffers, selected by sizeof storage.\n    alignas(16) uint8_t fetch_budget[147968];\n','')
index=helpers.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void qks_load_q(')
helpers=helpers[:index]+rename(function(control,'qkpv_write_o'))+rename(function(control,'qkpv_load_query'))+helpers[index:]
index=helpers.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void qks_produce(')
helpers=helpers[:index]+(p/'service.inc').read_text()+'\n'+helpers[index:]
old='    const uint32_t* scratch, uint32_t row, uint32_t kv_len) {'
new='    const uint32_t* scratch, uint32_t row, uint32_t kv_len,\n    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx, uint32_t epoch, uint32_t count) {'
assert helpers.count(old)==1;helpers=helpers.replace(old,new)
for expr in ['tile','next_hi','next_lo']:
 old=f'wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, {expr} * 32);'
 new=f'qks_wait_tile<KTraits>(params, storage, metadata, ctx, scratch, index_base, kv_len, {expr} * 32, epoch, count);'
 assert helpers.count(old)==1;helpers=helpers.replace(old,new)
key='    qks_all_sync();  // P and scale are ready; all KPE readers have finished.'
assert helpers.count(key)==1
helpers=helpers.replace(key,key+'\n    qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count);  // One subjob during PV-hi.')
start=helpers.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void qks_run_host_items(')
helpers=helpers[:start]+'''template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void qks_run_host_items(
    const Params& params, typename KTraits::SharedStorage& storage, uint8_t* smem,
    uint32_t* scratch, bool fetcher, uint32_t unique) {
  static_assert(Params::whole_rows && !ROW_MERGE);
  __shared__ QKSMetadata<typename Params::IdType> metadata;
  const uint32_t epoch = scratch[kHostBatchScratchEpoch];
  const uint32_t count = min(unique, params.staging_capacity);
  if (threadIdx.z == 0) {
    const uint32_t tid = threadIdx.y * 32 + threadIdx.x;
    QKSContext ctx;
    FAR_EMU_INIT(3);
    qks_producer_sync();
    if (qks_service_base(0) < count) qks_service_issue<0, KTraits>(params, storage, metadata, ctx, 0, count);
    if (qks_service_base(1) < count) qks_service_issue<1, KTraits>(params, storage, metadata, ctx, 1, count);
    if (qks_service_base(2) < count) qks_service_issue<2, KTraits>(params, storage, metadata, ctx, 2, count);
#pragma unroll 1
    for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
      qks_produce<KTraits>(params, storage, scratch, row, host_batch_row_len(params, row), metadata, ctx, epoch, count);
      qks_all_sync();  // Output alias is released before the next query load.
    }
#pragma unroll 1
    while (qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count)) {}
    cp_async::wait_group<0>();
    for (uint32_t slot = count + blockIdx.y * 128 + tid; slot < unique; slot += gridDim.y * 128)
      params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
    __threadfence();qks_producer_sync();
  } else {
#pragma unroll 1
    for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
      qks_consume<KTraits>(params, storage, row, host_batch_row_len(params, row));
      qks_all_sync();
    }
  }
}
'''
kernel=rename(function(control,'BatchMLAQKPVKernel')).replace('BatchMLAQKPVKernel','BatchMLAQKSServiceKernel')
# rename() already turns BatchMLAQKPVKernel into BatchMLAQKSKernel.
kernel=kernel.replace('BatchMLAQKSKernel','BatchMLAQKSServiceKernel')
kernel=kernel.replace('  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);','''  if (unique > 0) {
#pragma unroll 1
    for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
      run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique);
      __syncthreads();
    }
    // Shared payload and metadata may be reused after every pool has published its victims.
    cg::this_grid().sync();
  }
  const uint32_t fetch_rows = 64;''')
line='  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);\n'
assert kernel.count(line)==1;kernel=kernel.replace(line,'')
insert=control.index('// partial_capacity bounds the item states')
s=control[:insert]+helpers+kernel+'\n'+control[insert:]
a=control.index('    using QKPVKernelTraits = QKPVTraits<');b=control.index('    using CachedPEKernelTraits = CachedPETraits<',a)
dispatch=control[a:b];new=rename(dispatch).replace('qks_roles','qks').replace('BatchMLAQKSKernel','BatchMLAQKSServiceKernel')
new=new.replace('static_assert(qks_smem == 147968);','static_assert(qks_smem == 158464);')
assert 'static_assert(qks_smem == 158464);' in new
s=s.replace(dispatch,new+dispatch,1)
for variant,host in [('base',base),('control_qp',control),('qks',s)]:
 for name,value in [(variant+'.cuh',host),(variant+'_mla.cuh',math)]:
  f=p/name
  if f.exists():assert f.read_text()==value,f
  else:f.write_text(value)
 d=p/('include_'+variant)
 if not d.exists():
  shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
 assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==host
 assert (d/'flashinfer/attention/mla.cuh').read_text()==math
for name in ['plans.cpp','experiment.json','resources.py','resources_gpu3.py','bench.py','bench_gpu3_native.py','compile.py','test_cofetch_progress.py']:
 shutil.copy2(prior/name,p/name)
for name in ['candidatepaths.py','test_candidate.py','stress_candidate.py']:
 s=(prior/name).read_text().replace("'qkpv_roles'","'qks'")
 (p/name).write_text(s)
(p/'.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n')
(p/'qks_generated.inc').write_text(helpers)
m=dict(base_sha256=sha(base),control_sha256=sha(control),candidate_sha256=sha((p/'qks.cuh').read_text()),math_sha256=sha(math),candidate_adopted=False,goal_complete=False,
 mechanism='Four QK warps progress three8row service stages(two shared,one fiveuint4/lane register intent), during each PV-hi and while hostflags pending. EightPVwarps retain128output registers. Same384threadsinglecooperativekernel,upfrontallpoolLRUandgridbarrier,all108CTAworkerswithstaticrows,drainservicebeforeexit.',
 policy_change='All108CTA*24rows=2592inflight,notoriginal24fetchCTA*128rows=3072. API f24 is retained input but no dedicatedCTA partition. UpfrontLRU removes cachefill wait/defer because every victim is ready before service. Attention row assignment is staticstrided108 instead of claims/stealing.',
 dynamic_shared_bytes=158464,compute_fields_bytes=140032,service_shared_bytes=18432,register_payload_uint4_per_producer_lane_upper_bound=5,
 original_codegen_required=dict(EMU=40,normal=80,base_header='control_qp1252'),
 math='Copied unchanged QP attention helpers except service-progress parameters/calls;same equation/quantizedP/PE->CKV/jointmax. QP already changesFP32 grouping vsCPE.',
 risks='Incrementalservice may serializeonFARwait,uniform168regcapmaystillspill,contextaddressescapingandupfrontLRU/staticrowtailmayoffsetreuse. Every original40/80word and normalcache/Graph/sanitizer gate precedes timing.')
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
