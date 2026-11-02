// Deduplicate a bounded group in shared memory before issuing global claims.
// Global claims still select one owner across groups and across cache pools.
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_shared_claim_rows(
    const Params& params, uint32_t stamp, uint8_t* smem) {
  using Id=typename Params::IdType;
  constexpr uint32_t THREADS=KTraits::NUM_THREADS, BATCH=8, HASH=16384, GROUP=4;
  const uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
  const uint32_t lane=tid%32,lower=(1u<<lane)-1;
  auto* keys=reinterpret_cast<uint32_t*>(smem);
  const uint32_t units=ceil_div(params.num_rows,GROUP);
  for(uint32_t unit=blockIdx.y;unit<units;unit+=gridDim.y) {
    const uint32_t end=min((unit+1)*GROUP,params.num_rows);
    uint32_t first=unit*GROUP;
    while(first<end) {
      const uint32_t pool=params.row_pool[first];
      uint32_t last=first+1;
      while(last<end && params.row_pool[last]==pool)++last;
      for(uint32_t j=tid;j<HASH;j+=THREADS)keys[j]=0xffffffffu;
      __syncthreads();
      for(uint32_t row=first;row<last;++row) {
        const uint32_t begin=params.kv_indptr[row],length=params.kv_indptr[row+1]-begin;
        for(uint32_t offset=0;offset<length;offset+=BATCH*THREADS) {
          Id pages[BATCH];
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            const uint32_t at=offset+i*THREADS+tid;
            pages[i]=at<length?params.kv_indices[begin+at]:kHostBatchNoEntry;
          }
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            if(pages[i]>=params.host_page_start) {
              const int32_t slot=params.resident[pages[i]-params.host_page_start];
              if(slot>=0) {params.slot_stamp[slot]=stamp;pages[i]=kHostBatchNoEntry;}
            }
          }
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            if(pages[i]>=params.host_page_start) {
              const uint32_t page=static_cast<uint32_t>(pages[i]);
              uint32_t at=(page*2654435761u)&(HASH-1);
              while(true) {
                const uint32_t old=atomicCAS(keys+at,0xffffffffu,page);
                if(old==0xffffffffu || old==page)break;
                at=(at+1)&(HASH-1);
              }
            }
          }
        }
      }
      __syncthreads();
      // Rank owners at the group's first row. The group's unused entry_victim range
      // holds their page IDs; resolve copies it after the global row-prefix scan.
      for(uint32_t offset=0;offset<HASH;offset+=BATCH*THREADS) {
        uint32_t pages[BATCH],owned[BATCH],masks[BATCH],total=0;
#pragma unroll
        for(uint32_t i=0;i<BATCH;++i)pages[i]=keys[offset+i*THREADS+tid];
#pragma unroll
        for(uint32_t i=0;i<BATCH;++i)
          owned[i]=pages[i]!=0xffffffffu?atomicCAS(params.host_slot+pages[i]-params.host_page_start,0u,1u):2u;
#pragma unroll
        for(uint32_t i=0;i<BATCH;++i) {
          masks[i]=__ballot_sync(0xffffffff,owned[i]==0);total+=__popc(masks[i]);
        }
        uint32_t rank=0;
        if(lane==0 && total)rank=atomicAdd(host_batch_row_claims(params)+first,total);
        rank=__shfl_sync(0xffffffff,rank,0);
#pragma unroll
        for(uint32_t i=0;i<BATCH;++i) {
          if(masks[i]>>lane&1) {
            const uint32_t local=rank+__popc(masks[i]&lower);
            params.entry_victim[params.kv_indptr[first]+local]=pages[i];
            atomicExch(params.host_slot+pages[i]-params.host_page_start,(first<<kHostBatchRankBits|local)+2);
          }
          rank+=__popc(masks[i]);
        }
      }
      __syncthreads();
      // Emit the original Host-prefix/HBM-suffix layout without reclaiming pages.
      for(uint32_t row=first;row<last;++row) {
        const uint32_t begin=params.kv_indptr[row],length=params.kv_indptr[row+1]-begin;
        for(uint32_t offset=0;offset<length;offset+=BATCH*THREADS) {
          Id pages[BATCH];uint32_t hm[BATCH],lm[BATCH],hn=0,ln=0;
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            const uint32_t at=offset+i*THREADS+tid;
            pages[i]=at<length?params.kv_indices[begin+at]:kHostBatchNoEntry;
          }
          bool dropped=false;
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            if(pages[i]<0 && pages[i]!=kHostBatchNoEntry) {pages[i]=kHostBatchNoEntry;dropped=true;}
            if(pages[i]>=params.host_page_start) {
              const int32_t slot=params.resident[pages[i]-params.host_page_start];
              if(slot>=0) {params.slot_stamp[slot]=stamp;pages[i]=host_batch_cached_entry(params,slot);}
            }
          }
          if(__any_sync(0xffffffff,dropped) && lane==0)params.state[kHostBatchGaps]=1;
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            hm[i]=__ballot_sync(0xffffffff,pages[i]>=params.host_page_start);
            lm[i]=__ballot_sync(0xffffffff,pages[i]!=kHostBatchNoEntry && pages[i]<params.host_page_start);
            hn+=__popc(hm[i]);ln+=__popc(lm[i]);
          }
          uint32_t h=0,l=0;
          if(lane==0) {
            if(hn)h=atomicAdd(params.state+kHostBatchHostCounts+row,hn);
            if(ln)l=atomicAdd(host_batch_local_cursors(params)+row,ln);
          }
          h=__shfl_sync(0xffffffff,h,0);l=__shfl_sync(0xffffffff,l,0);
#pragma unroll
          for(uint32_t i=0;i<BATCH;++i) {
            if(pages[i]>=params.host_page_start)params.remap[begin+h+__popc(hm[i]&lower)]=pages[i];
            else if(pages[i]!=kHostBatchNoEntry)params.remap[begin+length-1-l-__popc(lm[i]&lower)]=pages[i];
            h+=__popc(hm[i]);l+=__popc(lm[i]);
          }
        }
      }
      __syncthreads();first=last;
    }
  }
}
