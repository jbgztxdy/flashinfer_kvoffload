// Compile-only structural feasibility. No kernel launch, no production edit.
#include <cstdint>
#include <cstdio>
#include "flashinfer/attention/mla.cuh"
using B=__nv_bfloat16;
using Traits=flashinfer::mla::KernelTraits<false,2,true,512,64,64,32,B,B,B,int>;
struct CompactAttention {
 union {
  struct {
   alignas(16) B q_smem_nope[64*512];
   alignas(16) B ckv_smem[2][32*512];
   alignas(16) B kpe_p_smem[2][32*64];
   union {alignas(16) float m_wg[2][64];alignas(16) float d_wg[2][64];};
  };
  alignas(16) B o_smem[64*512];
 };
};
struct UnifiedLayout {
 CompactAttention attention;
 alignas(128) B remote[2][10][576];
};
static_assert(sizeof(Traits::SharedStorage)==147968);
static_assert(sizeof(CompactAttention)==139776);
static_assert(sizeof(UnifiedLayout)==162816);
static_assert(sizeof(UnifiedLayout)+2304==165120);
static_assert(sizeof(UnifiedLayout)+2304<=166912);
int main(){printf("{\"classification\":\"CPUsizeofonly_not_real_kernel_resources\",\"original_dynamic\":%zu,\"compact_attention\":%zu,\"unified_dynamic\":%zu,\"assumed_static\":2304,\"inflight_rows\":2160,\"inflight_bytes\":2488320,\"BDP100GB20us_bytes\":2000000}\n",sizeof(Traits::SharedStorage),sizeof(CompactAttention),sizeof(UnifiedLayout));}
