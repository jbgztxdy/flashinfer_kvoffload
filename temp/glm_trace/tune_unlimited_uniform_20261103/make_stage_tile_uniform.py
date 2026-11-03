from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'retained.cuh').read_text()
a='''template <typename KTraits, typename Params>
__device__ __forceinline__ void load_host_batch_kv('''
b='''template <typename KTraits, typename Params, bool STAGED_ONLY = false>
__device__ __forceinline__ void load_host_batch_kv('''
assert s.count(a)==1;s=s.replace(a,b)
a='''    resolve_host_batch_row<KTraits>(params, pages[j], ckv_ptr, kpe_ptr);'''
b='''    if constexpr (STAGED_ONLY) {
      static_assert(Params::fully_staged);
      constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
      const int64_t slot = -static_cast<int64_t>(pages[j]) - 1;
      ckv_ptr = params.staging + slot * ROW + (lane_idx % 8) * upcast_size<DTypeKV>();
      kpe_ptr = ckv_ptr + KTraits::HEAD_DIM_CKV;
    } else {
      resolve_host_batch_row<KTraits>(params, pages[j], ckv_ptr, kpe_ptr);
    }'''
assert s.count(a)==1;s=s.replace(a,b)
start=s.index('__device__ __forceinline__ void pv_feature_attend_joint_host_tiles(')
end=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>',start)
body=s[start:end]
for tile in ['tile','next_hi','next_lo']:
 a=f'      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, {tile}*32, {tile}%2, index_base);'
 b=chr(10).join([
  '      // Every tile before the last host tile lies wholly in the staged prefix.',
  f'      if ({tile}+1 < host_tiles)',
  f'        load_host_batch_kv<KTraits, Params, true>(&storage, params, pages, kv_len, {tile}*32, {tile}%2, index_base);',
  '      else',
  f'        load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, {tile}*32, {tile}%2, index_base);'
 ])
 assert body.count(a)==1,(tile,body.count(a));body=body.replace(a,b)
s=s[:start]+body+s[end:]
name='stage_tile_uniform';assert not(p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_retained',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for kind in ['test','bench','run']:
 (p/(kind+'_'+name+'.py')).write_text((p/(kind+'_clean.py')).read_text().replace('clean',name))
f=p/('bench_'+name+'.py');t=f.read_text().replace("if VARIANT not in ['base','stage_tile_uniform']:","if VARIANT not in ['base','stage_tile_uniform']:");f.write_text(t)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'retained.cuh').read_bytes()).hexdigest(),mechanism='production PVFeature staged only: full32KV tiles before last host tile use branch-free negative staging slot pointer, boundary/local tiles original generic loader; fetch/LRU/bytes/math/order unchanged'),indent=2)+chr(10))
print(name,hashlib.sha256(s.encode()).hexdigest())
