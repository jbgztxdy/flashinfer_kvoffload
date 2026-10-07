# Purpose: Add the QK-only Host partition and scalar PV experiment.
# Usage: python temp/build_host_overlap_v30.py
from pathlib import Path
p = Path('include/flashinfer/attention/mla.cuh')
s = p.read_text().replace('    uint32_t count;\n    unsigned long long epoch;', '    uint32_t count;\n    uint32_t total_host;\n    unsigned long long epoch;')
s = s.replace('if (lane == 0 && warp == 0) storage->count = length;', 'if (lane == 0 && warp == 0) { storage->count = length; storage->total_host = total_host; }')
a = s.index('      if constexpr (!KTraits::QK_SHARD) {', s.index('void compute_host_split'))
s = s[:a]+'''      if constexpr (!KTraits::QK_SHARD) {
        if (storage.total_host <= 448) {
          const uint32_t host = blockIdx.y - 16;
          auto* host_m = params.partial_lse + 44 * 64;
          auto* host_d = host_m + 28 * 64;
          auto* host_ids = reinterpret_cast<int32_t*>(host_d + 28 * 64);
          const uint32_t tid = threadIdx.z * 128 + threadIdx.y * 32 + threadIdx.x;
          if (tid < 16) host_ids[host * 16 + tid] = tid < length ? storage.indices[tid] : 0;
          if (threadIdx.z == 0) {
            compute_mla_qk<KTraits>(&storage, 0, s_frag);
            logits_mask_<KTraits>(0, 0, 1, length, length, params.num_heads, s_frag);
            update_mdo_states_<KTraits>(&storage, 0, variant, s_frag, o_frag, m, d);
            typename KTraits::DTypeKV probabilities[8];
            vec_cast<typename KTraits::DTypeKV, float>::template cast<8>(probabilities, s_frag[0]);
            mma::m16k16_rowsum_f16f16f32(d, probabilities);
            finalize_m_<KTraits>(variant, m);
            const uint32_t lane = threadIdx.x, warp = threadIdx.y;
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              const uint32_t head = warp * 16 + lane / 4 + j * 8;
              if (lane % 4 == 0) {
                host_m[host * 64 + head] = m[j];
                host_d[host * 64 + head] = d[j];
              }
            }
            auto* destination = params.partial_o + 16 * 64 * 512 + host * 64 * 16;
#pragma unroll
            for (uint32_t reg = 0; reg < 8; ++reg) {
              const uint32_t head = warp * 16 + lane / 4 + ((reg % 4) / 2) * 8;
              const uint32_t key = 2 * (lane % 4) + (reg / 4) * 8 + reg % 2;
              destination[head * 16 + key] = probabilities[reg];
            }
          }
          return;
        }
      }
'''+s[a:]
a = s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void merge_ready_host_states')
b = s.index('template <typename KTraits, typename Params>\n__global__',a)
s = s[:a]+'''template <typename KTraits, typename Params>
__device__ __forceinline__ void merge_ready_host_states(
    typename KTraits::SharedStorage& storage, const Params& params) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, head = blockIdx.y - 44;
  float output[4], lse;
  merge_host_group<0, 16, KTraits>(storage, params, output, lse);
  const uint32_t total = params.host_prefetch_state[5624];
  float denominator = 1.f;
  if (total > 448) {
    float host[4], host_lse;
    merge_host_group<16, 28, KTraits>(storage, params, host, host_lse);
    const float maximum = max(lse, host_lse);
    const float local_weight = math::ptx_exp2(lse - maximum);
    const float host_weight = math::ptx_exp2(host_lse - maximum);
    denominator = local_weight + host_weight;
#pragma unroll
    for (uint32_t feature = 0; feature < 4; ++feature) output[feature] = output[feature] * local_weight + host[feature] * host_weight;
    lse = maximum;
  } else if (total > 0) {
    auto* host_m = params.partial_lse + 44 * 64;
    auto* host_d = host_m + 28 * 64;
    auto* host_ids = reinterpret_cast<const int32_t*>(host_d + 28 * 64);
    auto* probabilities = params.partial_o + 16 * 64 * 512;
    for (uint32_t part = 0; part < 28; ++part) {
      if (tid == 0) {
        unsigned long long ready;
        do {
          asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(ready) : "l"(params.host_prefetch_state + gridDim.y + 2048 + 16 + part) : "memory");
          if (ready != storage.epoch) __nanosleep(64);
        } while (ready != storage.epoch);
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ready) : "l"(params.host_prefetch_state + gridDim.y + 2048 + 16 + part) : "memory");
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      const float mass = host_d[part * 64 + head];
      if (mass > 0.f) {
        const float maximum = max(lse, host_m[part * 64 + head]);
        const float old_scale = math::ptx_exp2(lse - maximum);
        const float new_scale = math::ptx_exp2(host_m[part * 64 + head] - maximum);
        denominator = denominator * old_scale + mass * new_scale;
#pragma unroll
        for (uint32_t feature = 0; feature < 4; ++feature) output[feature] *= old_scale;
        lse = maximum;
        uint4 ids[4], packed_probabilities[2];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(ids[i].x), "=r"(ids[i].y), "=r"(ids[i].z), "=r"(ids[i].w) : "l"(host_ids + part * 16 + i * 4) : "memory");
        }
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
          asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(packed_probabilities[i].x), "=r"(packed_probabilities[i].y), "=r"(packed_probabilities[i].z), "=r"(packed_probabilities[i].w) : "l"(probabilities + (part * 64 + head) * 16 + i * 8) : "memory");
        }
        uint2 values[16];
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
          const uint32_t position = reinterpret_cast<const uint32_t*>(ids)[i];
          asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(values[i].x), "=r"(values[i].y) : "l"(params.host_prefetch_buffer + position * 576 + tid * 4) : "memory");
        }
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
          const float weight = float(reinterpret_cast<const typename KTraits::DTypeKV*>(packed_probabilities)[i]) * new_scale;
          if (weight > 0.f) {
            float value[4];
            vec_cast<float, typename KTraits::DTypeKV>::template cast<4>(value, reinterpret_cast<const typename KTraits::DTypeKV*>(values + i));
#pragma unroll
            for (uint32_t feature = 0; feature < 4; ++feature) output[feature] = fmaf(value[feature], weight, output[feature]);
          }
        }
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
    }
  }
#pragma unroll
  for (uint32_t feature = 0; feature < 4; ++feature) {
    params.final_o[head * params.o_stride_h + tid * 4 + feature] = typename KTraits::DTypeO(output[feature] / denominator);
  }
  if (tid == 0 && params.final_lse) {
    float value = lse + math::ptx_log2(denominator);
    if (params.return_lse_base_on_e) value *= math::loge2;
    params.final_lse[head] = value;
  }
}

'''+s[b:]
s = s.replace('''    const uint32_t length = prepare_host_split<KTraits>(&storage, params);
    record''','''    const uint32_t length = prepare_host_split<KTraits>(&storage, params);
    if (tid == 0 && blockIdx.y == 0) params.host_prefetch_state[5624] = storage.total_host;
    record''')
s = s.replace('''      if (tid < 64) params.partial_lse[blockIdx.y * 64 + tid] = -math::inf;''','''      if (tid < 64) {
        params.partial_lse[blockIdx.y * 64 + tid] = -math::inf;
        if (blockIdx.y >= 16 && storage.total_host <= 448) {
          auto* host_m = params.partial_lse + 44 * 64;
          auto* host_d = host_m + 28 * 64;
          host_m[(blockIdx.y - 16) * 64 + tid] = -math::inf;
          host_d[(blockIdx.y - 16) * 64 + tid] = 0.f;
        }
      }''')
p.write_text(s)
p = Path('flashinfer/mla/_core.py');s=p.read_text().replace('4096, dtype=torch.int64','8192, dtype=torch.int64');p.write_text(s)
