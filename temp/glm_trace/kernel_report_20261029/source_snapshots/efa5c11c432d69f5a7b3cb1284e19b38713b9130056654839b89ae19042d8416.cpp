// Offline exact per-launch LRU for the Memcpy data path. CPU work is outside GPU timing.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

extern "C" int64_t make_plan(const int32_t* pages, int32_t* remap, int64_t* src, int64_t* dst,
                              int64_t* offsets, int64_t* counts, int steps, int requests, int mtp,
                              int topk, int context, int near, int capacity) {
  struct Pool {
    std::unordered_map<int32_t,int> map;
    std::vector<int32_t> page, stamp;
    explicit Pool(int cap): page(cap,-1), stamp(cap,-1) {map.reserve(cap*2);}
  };
  std::vector<Pool> pools;
  for(int r=0;r<requests;r++) pools.emplace_back(capacity);
  int64_t used=0; offsets[0]=0;
  const int64_t entries=int64_t(requests)*mtp*topk;
  for(int t=0;t<steps;t++) {
    int64_t refs=0, hits=0, unique=0, misses=0;
    int overflow=0;
    for(int r=0;r<requests;r++) {
      auto& pool=pools[r];
      const int64_t begin=int64_t(t)*entries+int64_t(r)*mtp*topk;
      std::vector<int32_t> need; need.reserve(mtp*topk);
      for(int j=0;j<mtp*topk;j++) {int32_t p=pages[begin+j];if(p>=near) {need.push_back(p);refs++;}}
      std::sort(need.begin(),need.end());need.erase(std::unique(need.begin(),need.end()),need.end());
      unique+=need.size();
      std::vector<int32_t> missing;
      std::unordered_map<int32_t,int32_t> loc;loc.reserve(need.size()*2);
      for(int32_t p:need) {
        auto it=pool.map.find(p);
        if(it==pool.map.end()) missing.push_back(p);
        else {pool.stamp[it->second]=t;loc[p]=near+r*capacity+it->second;}
      }
      std::vector<int> victims;victims.reserve(capacity);
      for(int i=0;i<capacity;i++) if(pool.stamp[i]!=t) victims.push_back(i);
      std::sort(victims.begin(),victims.end(),[&](int a,int b){return pool.stamp[a]!=pool.stamp[b]?pool.stamp[a]<pool.stamp[b]:a<b;});
      for(size_t j=0;j<missing.size();j++) {
        int32_t p=missing[j];int32_t destination;
        if(j<victims.size()) {
          int slot=victims[j];if(pool.page[slot]>=0)pool.map.erase(pool.page[slot]);
          pool.page[slot]=p;pool.stamp[slot]=t;pool.map[p]=slot;destination=near+r*capacity+slot;
        } else destination=near+requests*capacity+overflow++;
        src[used]=p-near;dst[used]=destination;used++;loc[p]=destination;
      }
      misses+=missing.size();
      for(int j=0;j<mtp*topk;j++) {
        int32_t p=pages[begin+j];remap[begin+j]=p<near?p:loc.at(p);
        if(p>=near && std::binary_search(missing.begin(),missing.end(),p)==false)hits++;
      }
    }
    offsets[t+1]=used;counts[t*4]=refs;counts[t*4+1]=unique;counts[t*4+2]=misses;counts[t*4+3]=hits;
  }
  return used;
}
