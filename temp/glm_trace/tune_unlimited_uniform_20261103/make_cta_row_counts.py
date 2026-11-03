from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
start=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void prepare_host_batch_rows(')
end=s.index('// prepare_host_batch_rows without staging',start)
helper=s[start:end].replace('void prepare_host_batch_rows(const Params& params, uint32_t stamp)', 'void prepare_cta_owned_row_counts(const Params& params, uint32_t stamp, uint8_t* smem)')
a='''  constexpr uint32_t PER_THREAD = 8;''';b=a+'''
  if(params.prepare_parts!=1 || params.prepare_segment>8*KTraits::NUM_THREADS){
    prepare_host_batch_rows<KTraits>(params,stamp);return;
  }
  constexpr uint32_t WARPS=KTraits::NUM_THREADS/32;
  uint32_t* totals=reinterpret_cast<uint32_t*>(smem);''';assert helper.count(a)==1;helper=helper.replace(a,b)
a='''      if (lane == 0) {
        if (fresh_total) first = atomicAdd(host_batch_row_claims(params) + row, fresh_total);
        if (host_total) host_first = atomicAdd(host_cursors + row, host_total);
        if (local_total) local_first = atomicAdd(local_cursors + row, local_total);
      }
      first = __shfl_sync(0xffffffff, first, 0);
      host_first = __shfl_sync(0xffffffff, host_first, 0);
      local_first = __shfl_sync(0xffffffff, local_first, 0);'''
b='''      const uint32_t warp=tid/32;
      if(lane==0){totals[warp]=fresh_total;totals[WARPS+warp]=host_total;totals[2*WARPS+warp]=local_total;}
      __syncthreads();
      if(warp==0){
        uint32_t f=lane<WARPS?totals[lane]:0,h=lane<WARPS?totals[WARPS+lane]:0,l=lane<WARPS?totals[2*WARPS+lane]:0;
        const uint32_t f0=f,h0=h,l0=l;
#pragma unroll
        for(uint32_t delta=1;delta<WARPS;delta*=2){
          uint32_t pf=__shfl_up_sync(0xffffffff,f,delta),ph=__shfl_up_sync(0xffffffff,h,delta),pl=__shfl_up_sync(0xffffffff,l,delta);
          if(lane>=delta){f+=pf;h+=ph;l+=pl;}
        }
        if(lane<WARPS){totals[lane]=f-f0;totals[WARPS+lane]=h-h0;totals[2*WARPS+lane]=l-l0;}
        if(lane==WARPS-1){host_batch_row_claims(params)[row]=f;host_cursors[row]=h;local_cursors[row]=l;}
      }
      __syncthreads();
      first=totals[warp];host_first=totals[WARPS+warp];local_first=totals[2*WARPS+warp];''';assert helper.count(a)==1;helper=helper.replace(a,b)
a='''    }
  }
}

''';assert helper.endswith(a);helper=helper[:-len(a)]+'''      // Every reader finished before the next row reuses the prefix scratch.
      __syncthreads();
'''+a
marker='template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLACleanServiceKernel('
assert s.count(marker)==1;s=s.replace(marker,helper+marker)
start=s.index(marker);end=s.index('\ntemplate ',start)
body=s[start:end];a='prepare_host_batch_rows<KTraits>(params, epoch + 1)';assert body.count(a)==2;body=body.replace(a,'prepare_cta_owned_row_counts<KTraits>(params, epoch + 1, smem)');s=s[:start]+body+s[end:]
assert not(p/'cta_row_counts.cuh').exists();(p/'cta_row_counts.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_cta_row_counts');(p/'include_cta_row_counts/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_cta_row_counts.py').write_text((p/f'{n}_rescale_skip.py').read_text().replace('rescale_skip','cta_row_counts'))
(p/'cta_row_counts_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='only singleCTA singlepass row preparation: aggregate3warp counts via shared prefix, directstore exact row totals instead24atomicAdds; allhostslotCAS/EXCH andLRU retained; fallback original for multipart/longsegments'),indent=2)+'\n')
