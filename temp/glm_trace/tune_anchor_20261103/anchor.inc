// The exponent anchor may exceed the maximum: exp(x-a)/sum(exp(x-a)) is invariant.
// Round the log2 anchor upward by eight exponent units; only rebase O/d when it changes.
template <bool PAIR, typename KTraits>
__device__ __forceinline__ void anchor_update_mdo(
    typename KTraits::SharedStorage* storage, typename KTraits::AttentionVariant variant,
    float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d, bool want_lse, float* exact_d) {
  const uint32_t lane=threadIdx.x,wg=threadIdx.z,warp=threadIdx.y;
  const float scale=variant.sm_scale_log2;
  float previous[2];
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    previous[j]=m[j];
    float value=-math::inf;
#pragma unroll
    for(uint32_t t=0;t<(PAIR?2:1);++t) {
      const float* s=t==0?s0[0]:s1[0];
      value=max(value,max(max(s[j*2],s[j*2+1]),max(s[j*2+4],s[j*2+5])));
    }
    value=max(value,math::shfl_xor_sync(value,2));
    value=max(value,math::shfl_xor_sync(value,1));
    if(lane%4==0)storage->m_wg[wg][warp*16+j*8+lane/4]=value;
  }
  __syncthreads();
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
    const float value=max(storage->m_wg[0][warp*16+j*8+lane/4],storage->m_wg[1][warp*16+j*8+lane/4]);
    const float next=ceilf(value*scale*0.125f)*8.f;
    m[j]=max(previous[j],next);
  }
  // A warp skips the entire 128-value output rescale once all its anchors are stable.
  if(__any_sync(0xffffffff,m[0]!=previous[0] || m[1]!=previous[1])) {
#pragma unroll
    for(uint32_t j=0;j<2;++j) {
      const float factor=math::ptx_exp2(previous[j]-m[j]);
      d[j]*=factor;
      if(want_lse)exact_d[j]*=factor;
#pragma unroll
      for(uint32_t feature=0;feature<KTraits::NUM_MMA_D_CKV/2;++feature) {
        o[feature][j*2]*=factor;o[feature][j*2+1]*=factor;
        o[feature][j*2+4]*=factor;o[feature][j*2+5]*=factor;
      }
    }
  }
#pragma unroll
  for(uint32_t j=0;j<2;++j) {
#pragma unroll
    for(uint32_t t=0;t<(PAIR?2:1);++t) {
      float* s=t==0?s0[0]:s1[0];
      s[j*2]=math::ptx_exp2(s[j*2]*scale-m[j]);
      s[j*2+1]=math::ptx_exp2(s[j*2+1]*scale-m[j]);
      s[j*2+4]=math::ptx_exp2(s[j*2+4]*scale-m[j]);
      s[j*2+5]=math::ptx_exp2(s[j*2+5]*scale-m[j]);
      if(want_lse)exact_d[j]+=(s[j*2]+s[j*2+1])+(s[j*2+4]+s[j*2+5]);
    }
  }
}

