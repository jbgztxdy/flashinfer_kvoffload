// Stream the query fragments through L1/L2 instead of keeping a shared Q matrix.
// Its 64KiB shared footprint becomes a second 64-row KV stage.
template <typename KTraits, typename Params>
__device__ __forceinline__ void global_query_compute_qk(
    typename KTraits::SharedStorage& storage, const Params& params, uint32_t row,
    uint32_t stage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  const uint32_t lane=threadIdx.x, warp=threadIdx.y, group=threadIdx.z;
  const uint32_t head=warp*16+lane/4;
  smem_t<KTraits::SWIZZLE_MODE_KPE> pe(storage.kpe_p_smem[stage]);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[stage]);
  alignas(16) uint32_t qf[4],kf[4];
#pragma unroll
  for(uint32_t dim=0;dim<4;++dim) {
#pragma unroll
    for(uint32_t sub=0;sub<2;++sub) {
      const uint32_t offset=pe.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
          group*32+sub*16+8*(lane/16)+lane%8,2*dim+(lane%16)/8);
      pe.ldmatrix_m8n8x4(offset,kf);
      if(dim==0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(scores[sub],qpe[dim],kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(scores[sub],qpe[dim],kf);
    }
  }
  const auto* q=params.q_nope+row*params.q_nope_stride_n;
  uint32_t offset=kv.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(
      group*32+8*(lane/16)+lane%8,(lane%16)/8);
  const uint32_t other=16*KTraits::UPCAST_STRIDE_CKV;
#pragma unroll 4
  for(uint32_t dim=0;dim<32;++dim) {
    const uint32_t col=16*dim+2*(lane%4);
    qf[0]=*reinterpret_cast<const uint32_t*>(q+head*params.q_nope_stride_h+col);
    qf[1]=*reinterpret_cast<const uint32_t*>(q+(head+8)*params.q_nope_stride_h+col);
    qf[2]=*reinterpret_cast<const uint32_t*>(q+head*params.q_nope_stride_h+col+8);
    qf[3]=*reinterpret_cast<const uint32_t*>(q+(head+8)*params.q_nope_stride_h+col+8);
    kv.ldmatrix_m8n8x4(offset,kf);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(scores[0],qf,kf);
    kv.ldmatrix_m8n8x4(offset+other,kf);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(scores[1],qf,kf);
    offset=kv.template advance_offset_by_column<2>(offset,dim);
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void global_query_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES==2 && KTraits::CTA_TILE_KV==64 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];global_query_load_query(params,row,qpe);
  const int hi=ceil_div(kv_len,64u)-1;
#pragma unroll
  for(int i=0;i<2;++i) {
    const int tile=hi-i;
    if(tile>=0) {
      if(tile<host_tiles)wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,tile*64);
      __syncthreads();read_host_batch_pages<KTraits>(params,index_base,kv_len,tile*64,pages);
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,tile*64,tile%2,index_base);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for(int tile=hi;tile>=0;--tile) {
    const int next=tile-2;
    if(next>=0)read_host_batch_pages<KTraits>(params,index_base,kv_len,next*64,pages);
    if(tile>0)cp_async::wait_group<1>();else cp_async::wait_group<0>();
    __syncthreads();
    global_query_compute_qk<KTraits>(storage,params,row,tile%2,qpe,scores);
    if(tile==hi)logits_mask_<KTraits>(0,tile*64,1,kv_len,kv_len,params.num_heads,scores);
    update_mdo_states_<KTraits>(&storage,tile%2,variant,scores,o,m,d);
    compute_mla_pv<KTraits>(&storage,tile%2,scores,d,o);
    if(next>=0) {
      if(next<host_tiles)wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,next*64);
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,next*64,next%2,index_base);
      cp_async::commit_group();
    }
  }
  cp_async::wait_group<0>();__syncthreads();
}
