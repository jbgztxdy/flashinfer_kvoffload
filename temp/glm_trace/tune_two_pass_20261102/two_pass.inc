// Write raw scores with head contiguous, retaining native MMA output mapping.
template <typename KTraits, typename Params>
__device__ __forceinline__ void two_pass_store_scores(
    const Params& params, uint32_t index_base, uint32_t kv_len, uint32_t kv_base,
    float (*scores)[8], float* maxima) {
  const uint32_t lane=threadIdx.x, warp=threadIdx.y, group=threadIdx.z;
  float* all=reinterpret_cast<float*>(params.partial_o);
#pragma unroll
  for (uint32_t j=0;j<2;++j) {
    const uint32_t head=warp*16+j*8+lane/4;
#pragma unroll
    for (uint32_t c=0;c<4;++c) {
      const uint32_t col=kv_base+group*16+(lane%4)*2+(c/2)*8+c%2;
      const float value=scores[0][j*2+(c/2)*4+c%2];
      if (col<kv_len) {
        all[(static_cast<int64_t>(index_base)+col)*64+head]=value;
        maxima[j]=max(maxima[j],value);
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void two_pass_load_p(
    const Params& params, uint32_t index_base, uint32_t kv_len, uint32_t kv_base,
    float (*scores)[8], const float* maxima, float scale) {
  const uint32_t lane=threadIdx.x, warp=threadIdx.y, group=threadIdx.z;
  const float* all=reinterpret_cast<const float*>(params.partial_o);
#pragma unroll
  for (uint32_t j=0;j<2;++j) {
#pragma unroll
    for (uint32_t c=0;c<4;++c) {
      const uint32_t col=kv_base+group*16+(lane%4)*2+(c/2)*8+c%2;
      const uint32_t head=warp*16+j*8+lane/4;
      const float value=col<kv_len?all[(static_cast<int64_t>(index_base)+col)*64+head]:-math::inf;
      scores[0][j*2+(c/2)*4+c%2]=math::ptx_exp2(value*scale-maxima[j]*scale);
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void two_pass_load_value(
    typename KTraits::SharedStorage* storage, const Params& params,
    const typename KTraits::IdType* pages, uint32_t kv_len, uint32_t kv_base, uint32_t stage) {
  using DTypeKV=typename KTraits::DTypeKV;
  smem_t<KTraits::SWIZZLE_MODE_CKV> values(storage->ckv_smem[stage]);
#pragma unroll
  for (uint32_t j=0;j<kHostBatchTileRows<KTraits>;++j) {
    const uint32_t row=host_batch_tile_row<KTraits>(j);
    DTypeKV *ckv,*kpe;resolve_host_batch_row<KTraits>(params,pages[j],ckv,kpe);
#pragma unroll
    for (uint32_t dim=0;dim<KTraits::NUM_MMA_D_CKV/4;++dim) {
      const uint32_t offset=values.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(row,8*dim+threadIdx.x%8);
      values.template load_128b_async<SharedMemFillMode::kFillZero>(offset,ckv,kv_base+row<kv_len);
      ckv+=8*upcast_size<DTypeKV>();
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void two_pass_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES==2 && KTraits::CTA_TILE_KV==32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];two_pass_load_query(params,row,qpe);
  m[0]=m[1]=-math::inf;
  int hi=ceil_div(kv_len,32u)-1;
#pragma unroll
  for (int i=0;i<2;++i) {
    const int tile=hi-i;
    if (tile>=0) {
      if (tile<host_tiles)wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,tile*32);
      __syncthreads();read_host_batch_pages<KTraits>(params,index_base,kv_len,tile*32,pages);
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,tile*32,tile%2,index_base);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (;hi>=1;hi-=2) {
    cp_async::wait_group<0>();__syncthreads();
    two_pass_compute_qk_pair<KTraits>(storage,hi%2,(hi-1)%2,qpe,scores,scores+1);
    two_pass_store_scores<KTraits>(params,index_base,kv_len,hi*32,scores,m);
    two_pass_store_scores<KTraits>(params,index_base,kv_len,(hi-1)*32,scores+1,m);
    __syncthreads();
#pragma unroll
    for (int i=0;i<2;++i) {
      const int tile=hi-2-i;
      if (tile>=0) {
        if (tile<host_tiles)wait_host_batch_tile<KTraits>(params,scratch,index_base,kv_len,tile*32);
        __syncthreads();read_host_batch_pages<KTraits>(params,index_base,kv_len,tile*32,pages);
        load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,tile*32,tile%2,index_base);
        cp_async::commit_group();
      }
    }
  }
  cp_async::wait_group<0>();__syncthreads();
  if (hi==0) {
    two_pass_compute_one<KTraits>(storage,qpe,scores);
    two_pass_store_scores<KTraits>(params,index_base,kv_len,0,scores,m);
  }
  const uint32_t lane=threadIdx.x,warp=threadIdx.y,group=threadIdx.z;
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    m[j]=max(m[j],math::shfl_xor_sync(m[j],2));
    m[j]=max(m[j],math::shfl_xor_sync(m[j],1));
    if(lane%4==0)storage.m_wg[group][warp*16+j*8+lane/4]=m[j];
  }
  __syncthreads();
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    m[j]=max(storage.m_wg[0][warp*16+j*8+lane/4],storage.m_wg[1][warp*16+j*8+lane/4]);
    d[j]=0.f;
  }
#pragma unroll
  for(uint32_t feature=0;feature<KTraits::NUM_MMA_D_CKV/2;++feature)
#pragma unroll
    for(uint32_t r=0;r<8;++r)o[feature][r]=0.f;
  // All this row's source pages have landed during QK, so no additional ready wait is needed.
  __threadfence_block();__syncthreads();
  hi=ceil_div(kv_len,32u)-1;
#pragma unroll
  for(int i=0;i<2;++i) {
    const int tile=hi-i;
    if(tile>=0) {
      read_host_batch_pages<KTraits>(params,index_base,kv_len,tile*32,pages);
      two_pass_load_value<KTraits>(&storage,params,pages,kv_len,tile*32,tile%2);cp_async::commit_group();
    }
  }
#pragma unroll 1
  for(int tile=hi;tile>=0;--tile) {
    if(tile>0)cp_async::wait_group<1>();else cp_async::wait_group<0>();
    __syncthreads();
    two_pass_load_p<KTraits>(params,index_base,kv_len,tile*32,scores,m,variant.sm_scale_log2);
    compute_mla_pv<KTraits>(&storage,tile%2,scores,d,o);
    const int next=tile-2;
    if(next>=0) {
      __syncthreads();read_host_batch_pages<KTraits>(params,index_base,kv_len,next*32,pages);
      two_pass_load_value<KTraits>(&storage,params,pages,kv_len,next*32,next%2);cp_async::commit_group();
    }
  }
  cp_async::wait_group<0>();__syncthreads();
}
