// Offline exact per-launch LRU for the Memcpy data path. CPU work is outside GPU timing.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <cstring>

extern "C" int64_t make_plan(const int32_t* pages, int32_t* remap, int64_t* src, int64_t* dst,
                              int64_t* offsets, int64_t* counts, int steps, int requests, int mtp,
                              int topk, int context, int near, int capacity) {
  struct Pool {
    std::vector<int32_t> map, loc, page, stamp;
    Pool(int cap,int ctx): map(ctx,-1),loc(ctx,-1),page(cap,-1),stamp(cap,-1) {}
  };
  std::vector<Pool> pools;
  for(int r=0;r<requests;r++) pools.emplace_back(capacity,context);
  int64_t used=0; offsets[0]=0;
  const int64_t entries=int64_t(requests)*mtp*topk;
  for(int t=0;t<steps;t++) {
    std::vector<int64_t> statistics(requests*4), lengths(requests), overflow_counts(requests);
    #pragma omp parallel for num_threads(8) schedule(static)
    for(int r=0;r<requests;r++) {
      int64_t refs=0,hits=0,unique=0,misses=0;
      int overflow=0;
      int64_t written=0;
      auto& pool=pools[r];
      const int64_t begin=int64_t(t)*entries+int64_t(r)*mtp*topk;
      std::vector<int32_t> need; need.reserve(mtp*topk);
      for(int j=0;j<mtp*topk;j++) {int32_t p=pages[begin+j];if(p>=near) {need.push_back(p);refs++;}}
      std::sort(need.begin(),need.end());need.erase(std::unique(need.begin(),need.end()),need.end());
      unique+=need.size();
      std::vector<int32_t> missing;
      const int32_t base_page=near+r*context;
      for(int32_t p:need) {
        int slot=pool.map[p-base_page];
        if(slot<0) missing.push_back(p);
        else {pool.stamp[slot]=t;pool.loc[p-base_page]=near+r*capacity+slot;}
      }
      // Stamps are integers in [-1,t]. Stable counting sort is exactly the
      // former (stamp, slot-index) order, without sorting 4096 slots every step.
      std::vector<int> bucket(t+2,0);
      for(int i=0;i<capacity;i++) if(pool.stamp[i]!=t) bucket[pool.stamp[i]+1]++;
      int total=0;
      for(int& count:bucket) {int n=count;count=total;total+=n;}
      std::vector<int> victims(total);
      for(int i=0;i<capacity;i++) if(pool.stamp[i]!=t) victims[bucket[pool.stamp[i]+1]++]=i;
      for(size_t j=0;j<missing.size();j++) {
        int32_t p=missing[j];int32_t destination;
        if(j<victims.size()) {
          int slot=victims[j];if(pool.page[slot]>=0)pool.map[pool.page[slot]-base_page]=-1;
          pool.page[slot]=p;pool.stamp[slot]=t;pool.map[p-base_page]=slot;destination=near+r*capacity+slot;
        } else destination=near+requests*capacity+r*mtp*topk+overflow++;
        src[begin+written]=p-near;dst[begin+written]=destination;written++;pool.loc[p-base_page]=destination;
      }
      misses+=missing.size();
      for(int j=0;j<mtp*topk;j++) {
        int32_t p=pages[begin+j];remap[begin+j]=p<near?p:pool.loc[p-base_page];
        if(p>=near && std::binary_search(missing.begin(),missing.end(),p)==false)hits++;
      }
      lengths[r]=written;overflow_counts[r]=overflow;
      statistics[r*4]=refs;statistics[r*4+1]=unique;statistics[r*4+2]=misses;statistics[r*4+3]=hits;
    }
    int overflow_prefix=0;
    for(int k=0;k<4;k++)counts[t*4+k]=0;
    for(int r=0;r<requests;r++) {
      int64_t begin=int64_t(t)*entries+int64_t(r)*mtp*topk;
      int delta=r*mtp*topk-overflow_prefix;
      for(int64_t j=0;j<lengths[r];j++)if(dst[begin+j]>=near+requests*capacity)dst[begin+j]-=delta;
      for(int j=0;j<mtp*topk;j++)if(remap[begin+j]>=near+requests*capacity)remap[begin+j]-=delta;
      std::memmove(src+used,src+begin,lengths[r]*sizeof(int64_t));
      std::memmove(dst+used,dst+begin,lengths[r]*sizeof(int64_t));
      used+=lengths[r];overflow_prefix+=overflow_counts[r];
      for(int k=0;k<4;k++)counts[t*4+k]+=statistics[r*4+k];
    }
    offsets[t+1]=used;
  }
  return used;
}
