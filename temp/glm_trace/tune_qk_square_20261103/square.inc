// Four warps compute a pair of tiles as four32x32 score blocks. Each Q fragment
// feeds two key blocks and each K fragment feeds two head blocks, all in one warp.
template <typename Params>
__device__ __forceinline__ void square_load_query(const Params& params,uint32_t row,
                                                 uint32_t (&qpe)[8][4]) {
  if(threadIdx.z!=0)return;
  const uint32_t lane=threadIdx.x,head=(threadIdx.y%2)*32+lane/4;
  const auto* q=params.q_pe+row*params.q_pe_stride_n;
#pragma unroll
  for(uint32_t headpart=0;headpart<2;++headpart)
#pragma unroll
    for(uint32_t dim=0;dim<4;++dim) {
      const uint32_t h=head+headpart*16,col=dim*16+(lane%4)*2;
      qpe[headpart*4+dim][0]=*reinterpret_cast<const uint32_t*>(q+h*params.q_pe_stride_h+col);
      qpe[headpart*4+dim][1]=*reinterpret_cast<const uint32_t*>(q+(h+8)*params.q_pe_stride_h+col);
      qpe[headpart*4+dim][2]=*reinterpret_cast<const uint32_t*>(q+h*params.q_pe_stride_h+col+8);
      qpe[headpart*4+dim][3]=*reinterpret_cast<const uint32_t*>(q+(h+8)*params.q_pe_stride_h+col+8);
    }
}

template <bool PE,typename KTraits>
__device__ __forceinline__ void square_dot(
    typename KTraits::SharedStorage& storage,uint32_t stage,
    uint32_t (&qpe)[8][4],float (&score)[4][8]) {
  const uint32_t lane=threadIdx.x,heads=threadIdx.y%2*32;
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[stage]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> pe(storage.kpe_p_smem[stage]);
  alignas(16) uint32_t qa[4],qb[4],ka[4],kb[4];
#pragma unroll 4
  for(uint32_t dim=0;dim<(PE?4:32);++dim) {
    if constexpr(PE) {
#pragma unroll
      for(uint32_t r=0;r<4;++r) {qa[r]=qpe[dim][r];qb[r]=qpe[4+dim][r];}
    } else {
      const uint32_t offset=q.template get_permuted_offset<KTraits::UPCAST_STRIDE_Q_NOPE>(heads+lane%16,2*dim+lane/16);
      q.ldmatrix_m8n8x4(offset,qa);
      q.ldmatrix_m8n8x4(offset+16*KTraits::UPCAST_STRIDE_Q_NOPE,qb);
    }
    if constexpr(PE) {
      const uint32_t offset=pe.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(8*(lane/16)+lane%8,2*dim+(lane%16)/8);
      pe.ldmatrix_m8n8x4(offset,ka);
      pe.ldmatrix_m8n8x4(offset+16*KTraits::UPCAST_STRIDE_KPE,kb);
    } else {
      const uint32_t offset=kv.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(8*(lane/16)+lane%8,2*dim+(lane%16)/8);
      kv.ldmatrix_m8n8x4(offset,ka);
      kv.ldmatrix_m8n8x4(offset+16*KTraits::UPCAST_STRIDE_CKV,kb);
    }
    if(PE && dim==0) {
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(score[0],qa,ka);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(score[1],qa,kb);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(score[2],qb,ka);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(score[3],qb,kb);
    } else {
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(score[0],qa,ka);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(score[1],qa,kb);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(score[2],qb,ka);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(score[3],qb,kb);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void square_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    uint32_t (&qpe)[8][4],float (*sa)[8],float (*sb)[8]) {
  const uint32_t lane=threadIdx.x,warp=threadIdx.y,group=threadIdx.z;
  if(group==0) {
    float score[4][8];const uint32_t tile=warp/2;
    square_dot<true,KTraits>(storage,tile==0?a:b,qpe,score);
    square_dot<false,KTraits>(storage,tile==0?a:b,qpe,score);
#pragma unroll
    for(uint32_t headpart=0;headpart<2;++headpart)
#pragma unroll
      for(uint32_t colpart=0;colpart<2;++colpart)
#pragma unroll
        for(uint32_t j=0;j<2;++j)
#pragma unroll
          for(uint32_t c=0;c<4;++c) {
            const uint32_t head=warp%2*32+headpart*16+j*8+lane/4;
            const uint32_t col=colpart*16+(lane%4)*2+c/2*8+c%2;
            storage.score_exchange[tile][head*32+col]=score[headpart*2+colpart][j*2+c/2*4+c%2];
          }
  }
  __syncthreads();
#pragma unroll
  for(uint32_t tile=0;tile<2;++tile)
#pragma unroll
    for(uint32_t j=0;j<2;++j)
#pragma unroll
      for(uint32_t c=0;c<4;++c) {
        const uint32_t head=warp*16+j*8+lane/4;
        const uint32_t col=group*16+(lane%4)*2+c/2*8+c%2;
        (tile==0?sa:sb)[0][j*2+c/2*4+c%2]=storage.score_exchange[tile][head*32+col];
      }
}
