// The square QK producer publishes quantized P directly in the original two
// KPE/P stages, avoiding an FP32-score round trip and both native P republish steps.
template <typename KTraits>
__device__ __forceinline__ void square_p_prepare(
    typename KTraits::SharedStorage& storage,typename KTraits::AttentionVariant variant,
    uint32_t a,uint32_t b,uint32_t kv_base_a,uint32_t kv_base_b,uint32_t kv_len,
    uint32_t (&qpe)[8][4],float (*o)[8],float* m,float* d) {
  const uint32_t lane=threadIdx.x,warp=threadIdx.y,group=threadIdx.z;
  const uint32_t tile=warp/2,heads=warp%2*32,stage=tile==0?a:b;
  float score[4][8];
  if(group==0) {
    square_dot<true,KTraits>(storage,stage,qpe,score);
    square_dot<false,KTraits>(storage,stage,qpe,score);
#pragma unroll
    for(uint32_t hp=0;hp<2;++hp)
#pragma unroll
      for(uint32_t j=0;j<2;++j) {
        float maximum=-math::inf;
#pragma unroll
        for(uint32_t cp=0;cp<2;++cp)
#pragma unroll
          for(uint32_t c=0;c<4;++c) {
            const uint32_t col=cp*16+(lane%4)*2+c/2*8+c%2;
            float& value=score[hp*2+cp][j*2+c/2*4+c%2];
            if((tile==0?kv_base_a:kv_base_b)+col>=kv_len)value=-math::inf;
            maximum=max(maximum,value);
          }
        maximum=max(maximum,math::shfl_xor_sync(maximum,2));
        maximum=max(maximum,math::shfl_xor_sync(maximum,1));
        if(lane%4==0)storage.m_wg[tile][heads+hp*16+j*8+lane/4]=maximum;
      }
  }
  __syncthreads();
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    const uint32_t head=warp*16+j*8+lane/4;
    const float previous=m[j];m[j]=max(m[j],max(storage.m_wg[0][head],storage.m_wg[1][head]));
    const float factor=math::ptx_exp2(previous*variant.sm_scale_log2-m[j]*variant.sm_scale_log2);
    d[j]*=factor;
#pragma unroll
    for(uint32_t f=0;f<KTraits::NUM_MMA_D_CKV/2;++f) {
      o[f][j*2]*=factor;o[f][j*2+1]*=factor;o[f][j*2+4]*=factor;o[f][j*2+5]*=factor;
    }
    if(group==0 && lane%4==0)storage.square_anchor[head]=m[j];
  }
  __syncthreads();
  if(group==0) {
    smem_t<KTraits::SWIZZLE_MODE_P> p(storage.kpe_p_smem[stage]);
#pragma unroll
    for(uint32_t hp=0;hp<2;++hp)
#pragma unroll
      for(uint32_t cp=0;cp<2;++cp) {
#pragma unroll
        for(uint32_t j=0;j<2;++j) {
          const float anchor=storage.square_anchor[heads+hp*16+j*8+lane/4];
#pragma unroll
          for(uint32_t c=0;c<4;++c) {
            float& value=score[hp*2+cp][j*2+c/2*4+c%2];
            value=math::ptx_exp2(value*variant.sm_scale_log2-anchor*variant.sm_scale_log2);
          }
        }
        alignas(16) typename KTraits::DTypeKV probability[8];
        vec_cast<typename KTraits::DTypeKV,float>::cast<8>(probability,score[hp*2+cp]);
        float total[2]={0.f,0.f};mma::m16k16_rowsum_f16f16f32(total,probability);
#pragma unroll
        for(uint32_t j=0;j<2;++j)
          if(lane%4==0)storage.square_sum[tile][cp][heads+hp*16+j*8+lane/4]=total[j];
        const uint32_t offset=p.template get_permuted_offset<KTraits::UPCAST_STRIDE_P>(heads+hp*16+lane/4,cp*2);
        ((uint32_t*)(p.base+offset))[lane%4]=*reinterpret_cast<uint32_t*>(&probability[0]);
        ((uint32_t*)(p.base+offset+8*KTraits::UPCAST_STRIDE_P))[lane%4]=*reinterpret_cast<uint32_t*>(&probability[2]);
        ((uint32_t*)(p.base+(offset^1)))[lane%4]=*reinterpret_cast<uint32_t*>(&probability[4]);
        ((uint32_t*)(p.base+(offset^1)+8*KTraits::UPCAST_STRIDE_P))[lane%4]=*reinterpret_cast<uint32_t*>(&probability[6]);
      }
  }
  __syncthreads();
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    const uint32_t head=warp*16+j*8+lane/4;
    d[j]+=storage.square_sum[0][group][head];
    d[j]+=storage.square_sum[1][group][head];
  }
}

template <typename KTraits>
__device__ __forceinline__ void square_p_pv(
    typename KTraits::SharedStorage& storage,uint32_t stage,float (*o)[8]) {
  const uint32_t lane=threadIdx.x,warp=threadIdx.y,group=threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[stage]);
  smem_t<KTraits::SWIZZLE_MODE_P> p(storage.kpe_p_smem[stage]);
  uint32_t ko=kv.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(lane%16,group*KTraits::NUM_MMA_D_CKV+lane/16);
  uint32_t po=p.template get_permuted_offset<KTraits::UPCAST_STRIDE_P>(warp*16+lane%16,lane/16);
#pragma unroll
  for(uint32_t k=0;k<KTraits::NUM_MMA_KV;++k) {
    uint32_t pf[4];p.ldmatrix_m8n8x4(po,pf);po=p.template advance_offset_by_column<2>(po,k);
#pragma unroll
    for(uint32_t f=0;f<KTraits::NUM_MMA_D_CKV/2;++f) {
      uint32_t vf[4];kv.ldmatrix_m8n8x4_trans(ko,vf);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o[f],pf,vf);
      ko=kv.template advance_offset_by_column<2>(ko,f);
    }
    ko=kv.template advance_offset_by_row<16,KTraits::UPCAST_STRIDE_CKV>(ko)-KTraits::NUM_MMA_D_CKV;
  }
}
