// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void host_batch_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    const uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
        warp*16+lane%16,dim*2+lane/16);
    q.ldmatrix_m8n8x4(qo,qf);
    const uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
        group*16+8*(lane/16)+lane%8,2*dim+(lane%16)/8);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
  }
}

template <typename KTraits>
__device__ __forceinline__ void host_batch_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    float (*sa)[8],float (*sb)[8]) {
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> qp(storage.q_smem_pe);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpa(storage.kpe_p_smem[a]),kpb(storage.kpe_p_smem[b]);
  host_batch_qk_pair_fragments<true,KTraits,KTraits::NUM_MMA_D_KPE,
      KTraits::UPCAST_STRIDE_Q_PE,KTraits::UPCAST_STRIDE_KPE>(qp,kpa,kpb,sa,sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  host_batch_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}

template <typename KTraits,typename Params>
__device__ __forceinline__ void attend_host_batch_tile_pairs(
    const Params& params,typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant,const uint32_t* scratch,
    uint32_t index_base,uint32_t kv_len,int host_tiles,float (*sa)[8],
    float (*o)[8],float* m,float* d) {
  static_assert(KTraits::NUM_STAGES==2 && KTraits::CTA_TILE_KV==32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  float sb[KTraits::NUM_MMA_KV/2][8];
  int tile=ceil_div(kv_len,32u)-1;
  const int mask_tile=kv_len/32;
#pragma unroll
  for (int i=0;i<2;++i) {
    const int t=tile-i;
    if (t>=0 && t<host_tiles) wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,t*32);
  }
  __syncthreads();
#pragma unroll
  for (int i=0;i<2;++i) {
    const int t=tile-i;
    if (t>=0) {
      read_host_batch_pages<KTraits>(params,index_base,kv_len,t*32,pages);
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,t*32,t%2,index_base);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (;tile>=0;tile-=2) {
    cp_async::wait_group<0>();
    __syncthreads();
    if (tile>0) host_batch_compute_qk_pair<KTraits>(storage,tile%2,(tile-1)%2,sa,sb);
    else compute_mla_qk<KTraits>(&storage,tile%2,sa);
    if (tile==mask_tile)
      logits_mask_<KTraits>(0,tile*32,1,kv_len,kv_len,params.num_heads,sa);
    update_mdo_states_<KTraits>(&storage,tile%2,variant,sa,o,m,d);
    compute_mla_pv<KTraits>(&storage,tile%2,sa,d,o);
    const int next_a=tile-2;
    if (next_a>=0) {
      if (next_a<host_tiles) wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,next_a*32);
      __syncthreads();
      read_host_batch_pages<KTraits>(params,index_base,kv_len,next_a*32,pages);
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,next_a*32,next_a%2,index_base);
      cp_async::commit_group();
    }
    if (tile>0) {
      update_mdo_states_<KTraits>(&storage,(tile-1)%2,variant,sb,o,m,d);
      compute_mla_pv<KTraits>(&storage,(tile-1)%2,sb,d,o);
      const int next_b=tile-3;
      if (next_b>=0) {
        if (next_b<host_tiles) wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,next_b*32);
        __syncthreads();
        read_host_batch_pages<KTraits>(params,index_base,kv_len,next_b*32,pages);
        load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,next_b*32,next_b%2,index_base);
        cp_async::commit_group();
      }
    }
  }
}
