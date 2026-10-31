#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <queue>
#include <vector>
using std::min; using std::max;
template <typename T> T ceil_div(T a,T b) { return (a+b-1)/b; }
template <typename T> T round_up(T a,T b) { return ceil_div(a,b)*b; }
// Host-only schedule estimate for the medium whole-row dispatch. The original tail planner
// charges six KV tiles for Q/partial/merge traffic; direct rows charge four (Q and final O).
// Model the existing two-ended cyclic item assignment with no remote stall. A short last
// wave of whole rows must not replace a faster split plan just because there are two SM waves.
struct HostBatchCostEvent {
  uint64_t finish;
  uint32_t cta;
  int32_t item;
};
struct HostBatchCostLater {
  bool operator()(const HostBatchCostEvent& a, const HostBatchCostEvent& b) const {
    return a.finish > b.finish || (a.finish == b.finish && a.cta > b.cta);
  }
};
static inline bool host_batch_balanced_whole_rows_uncached(uint32_t rows, uint32_t max_kv_len,
                                          uint32_t grid, uint32_t fetch,
                                          uint32_t partial_capacity) {
  constexpr uint32_t TILE = 32;
  if (grid == 0 || fetch == 0 || fetch >= grid) return false;
  const uint32_t workers = grid - fetch;
  const uint32_t per_worker = rows / grid > 0 ? rows / grid - 1 : 0;
  const uint32_t full = min(rows, workers * per_worker);
  const uint32_t tail = rows - full;
  if (2 * tail > partial_capacity || tail == 0) return true;
  const uint32_t length = max(max_kv_len, 1u);
  uint32_t chunks = 1;
  uint64_t best = UINT64_MAX;
  for (uint32_t c = 1; c <= 32; ++c) {
    const uint32_t chunk_len = round_up(ceil_div(length, c), TILE);
    if (c > 1 && tail * (c + 1) > partial_capacity) break;
    if (c > 1 && chunk_len * (c - 1) >= length) continue;
    const uint64_t cost = ceil_div(tail * c, grid) * (chunk_len / TILE + 6u);
    if (cost < best) { best = cost; chunks = c; }
  }
  const uint32_t chunk_len = round_up(ceil_div(length, chunks), TILE);
  auto predict = [&](bool whole) {
    const uint32_t items = whole ? rows : full + tail * chunks;
    std::vector<uint8_t> claimed(items, 0);
    std::priority_queue<HostBatchCostEvent, std::vector<HostBatchCostEvent>,
                        HostBatchCostLater> pending;
    for (uint32_t cta = 0; cta < grid; ++cta) {
      const int32_t first = cta < workers ? static_cast<int32_t>(cta)
          : static_cast<int32_t>(items) - 1 - static_cast<int32_t>(cta - workers);
      pending.push({0, cta, first});
    }
    uint64_t finish = 0;
    while (!pending.empty()) {
      auto event = pending.top(); pending.pop();
      const bool back = event.cta >= workers;
      const int32_t step = back ? -static_cast<int32_t>(fetch)
                               : static_cast<int32_t>(workers);
      while (event.item >= 0 && static_cast<uint32_t>(event.item) < items &&
             claimed[event.item]) {
        if (back) { event.item = -1; break; }
        event.item += step;
      }
      if (event.item < 0 || static_cast<uint32_t>(event.item) >= items) continue;
      claimed[event.item] = 1;
      uint32_t cost = ceil_div(length, TILE) + 4u;
      if (!whole && static_cast<uint32_t>(event.item) >= full) {
        const uint32_t chunk = (static_cast<uint32_t>(event.item) - full) % chunks;
        cost = ceil_div(min(chunk_len, length - chunk * chunk_len), TILE) + 6u;
      }
      event.finish += cost;
      finish = event.finish > finish ? event.finish : finish;
      event.item += step;
      pending.push(event);
    }
    return finish;
  };
  return predict(true) <= predict(false);
}


// One bounded host memo avoids re-running shape planning on a stable decode shape. It
// depends only on the five scalar inputs, never on KV contents, device state or link timing.
struct HostBatchShapeMemo {
  uint32_t rows, length, grid, fetch, capacity;
  bool valid, whole;
};
static inline bool host_batch_balanced_whole_rows(uint32_t rows, uint32_t max_kv_len,
                                                 uint32_t grid, uint32_t fetch,
                                                 uint32_t partial_capacity) {
  static thread_local HostBatchShapeMemo last{};
  if (last.valid && last.rows == rows && last.length == max_kv_len && last.grid == grid &&
      last.fetch == fetch && last.capacity == partial_capacity) return last.whole;
  const bool whole = host_batch_balanced_whole_rows_uncached(rows, max_kv_len, grid, fetch,
                                                            partial_capacity);
  last = {rows, max_kv_len, grid, fetch, partial_capacity, true, whole};
  return whole;
}


int main() {
  unsigned checks=0;
  for (uint32_t rows:{216u,224u,256u,320u,384u})
  for (uint32_t length:{97u,2048u})
  for (uint32_t grid:{80u,108u,132u})
  for (uint32_t fetch:{8u,16u,24u})
  for (uint32_t capacity:{1u,864u}) {
    bool expected=host_batch_balanced_whole_rows_uncached(rows,length,grid,fetch,capacity);
    assert(host_batch_balanced_whole_rows(rows,length,grid,fetch,capacity)==expected);
    assert(host_batch_balanced_whole_rows(rows,length,grid,fetch,capacity)==expected);
    checks+=2;
  }
  auto start=std::chrono::steady_clock::now();
  volatile bool selected=host_batch_balanced_whole_rows(256,2048,108,16,864);
  double cold=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  start=std::chrono::steady_clock::now();
  for(int i=0;i<1000000;++i) {
    asm volatile("" ::: "memory");
    selected=host_batch_balanced_whole_rows(256,2048,108,16,864);
  }
  double warm=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/1000000;
  assert(selected);
  assert(!host_batch_balanced_whole_rows(224,2048,108,16,864));
  std::cout << "{\"memo_equivalence_checks\":" << checks
    << ",\"cold_us\":" << cold << ",\"warm_us_per_call\":" << warm << "}\n";
}
