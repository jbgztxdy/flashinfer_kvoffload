"""One-time warp flat-combining of already-issued requests, exact max-plus costs."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent
assert not(p/'batch.cuh').exists()
s=(p/'base.cuh').read_text();at=s.index('// Per CTA/stage,')
helper=r'''
// Isolated model candidate. Only already-issued requests enter these mailboxes.
// A winning requester warp combines all currently published mailboxes; there is
// no dedicated CTA/SM, no future byte reservation and no fixed perCTA bandwidth.
constexpr uint32_t kBrokerCTAs=8192,kBrokerWarps=8;
struct BrokerRequest {unsigned long long issue,cost,deadline,record;};
__device__ BrokerRequest broker_requests[kBrokerCTAs*kBrokerWarps];
__device__ uint32_t broker_masks[kBrokerCTAs],broker_lock;
struct BrokerRecord {unsigned long long issue,cost,prior,end,consume,queue;};
constexpr uint32_t kBrokerRecords=100000;
__device__ BrokerRecord broker_records[kBrokerRecords];
__device__ unsigned long long broker_count;
__device__ uint32_t broker_trace;
__device__ __forceinline__ uint32_t broker_acquire(const uint32_t* p){uint32_t v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ unsigned long long broker_acquire64(const unsigned long long* p){unsigned long long v;asm volatile("ld.acquire.gpu.global.u64 %0,[%1];":"=l"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void broker_release(uint32_t* p,uint32_t v){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ void broker_release64(unsigned long long* p,unsigned long long v){asm volatile("st.release.gpu.global.u64 [%0],%1;"::"l"(p),"l"(v):"memory");}
__device__ __forceinline__ unsigned long long broker_shfl(unsigned long long v,int lane){return __shfl_sync(0xffffffffu,v,lane);}
__device__ __forceinline__ void broker_combine(){
 const uint32_t lane=threadIdx.x%32,ctas=gridDim.x*gridDim.y*gridDim.z;
 unsigned long long clock=lane==0?link_clock:0;clock=broker_shfl(clock,0);
 unsigned long long record=lane==0?broker_count:0;record=broker_shfl(record,0);
 for(uint32_t base=0;base<ctas;base+=32){
  uint32_t c=base+lane,mask=c<ctas?broker_acquire(broker_masks+c):0;
  uint32_t any=mask;
#pragma unroll
  for(uint32_t d=16;d>0;d/=2)any|=__shfl_xor_sync(0xffffffffu,any,d);
  while(any){
   uint32_t w=__ffs(any)-1;any&=any-1;bool active=(mask>>w)&1u;
   auto* q=broker_requests+c*kBrokerWarps+w;
   unsigned long long cost=active?q->cost:0,issue=active?q->issue:0,prefix=cost;
#pragma unroll
   for(uint32_t d=1;d<32;d*=2){auto a=__shfl_up_sync(0xffffffffu,prefix,d);if(lane>=d)prefix+=a;}
   // Prefix max-plus scan: end_i=prefixCost_i+max(clock,max_j<=i(issue_j-prefixCostBefore_j)).
   long long anchor=active?static_cast<long long>(issue)-static_cast<long long>(prefix-cost):(-0x7fffffffffffffffll);
   anchor=max(anchor,static_cast<long long>(clock));
#pragma unroll
   for(uint32_t d=1;d<32;d*=2){auto a=__shfl_up_sync(0xffffffffu,anchor,d);if(lane>=d)anchor=max(anchor,a);}
   unsigned long long end=static_cast<unsigned long long>(anchor)+prefix;
   const uint32_t votes=__ballot_sync(0xffffffffu,active),below=__popc(votes&((1u<<lane)-1));
   const unsigned long long id=record+below;
   if(active){
    // Remove this request before publishing its answer. The producer may reuse
    // its mailbox only after acquiring that answer, so the next bit cannot be cleared here.
    atomicAnd(broker_masks+c,~(1u<<w));
    if(broker_trace&&id<kBrokerRecords)broker_records[id]={issue,cost,end-cost,end,0,c*kBrokerWarps+w};
    q->record=id;
    broker_release64(&q->deadline,end+config.latency_ns);
   }
   clock=broker_shfl(end,31);record+=__popc(votes);
  }
 }
 if(lane==0){link_clock=clock;broker_count=record;broker_release(&broker_lock,0);}
}
// Every lane in the loading warp calls this function. The request is published
// after real copies have been issued, and receives exactly its own service cost.
__device__ __forceinline__ unsigned long long reserve_warp(unsigned long long bytes){
 if(!bytes)return 0;
 const uint32_t lane=threadIdx.x%32,t=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
 const uint32_t c=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z),w=t/32;
 auto* q=broker_requests+c*kBrokerWarps+w;
 if(lane==0){
  q->deadline=0;q->issue=now_ns();q->cost=(bytes*config.ps_per_byte+999)/1000;
  __threadfence();atomicOr(broker_masks+c,1u<<w);
 }
 __syncwarp();
 unsigned long long answer=0;
 while(true){
  if(lane==0)answer=broker_acquire64(&q->deadline);answer=broker_shfl(answer,0);
  if(answer)return answer;
  uint32_t won=lane==0?atomicCAS(&broker_lock,0u,1u):0;won=__shfl_sync(0xffffffffu,won,0);
  if(won==0)broker_combine();else __nanosleep(64);
 }
}
'''
s=s[:at]+helper+s[at:]
a=s.index('  if (threadIdx.x % 32 == 0 && rows) {',s.index('void tile('));b=s.index('\n}',a)
s=s[:a]+'''  if(rows){
    unsigned long long deadline;
    if(config.ps_per_byte==0){deadline=now_ns()+config.latency_ns;deadline=broker_shfl(deadline,0);}
    else deadline=reserve_warp(static_cast<unsigned long long>(rows)*row_bytes);
    if(threadIdx.x%32==0)atomicMax(&pending_ready[cta_slot()][stage],deadline);
  }'''+s[b:]
(p/'batch.cuh').write_text(s);(p/'include/flashinfer/attention/far_link_emu.cuh').write_text(s)
s=(p.parent/'tune_uniform_warps_20261103/transfer_modes.cu').read_text()
s=s.replace('flashinfer::far_emu::link_clock=0;','{flashinfer::far_emu::link_clock=0;flashinfer::far_emu::broker_lock=0;flashinfer::far_emu::broker_count=0;}')
(p/'batch.cu').write_text(s);(p/'run_batch.py').write_text((p/'run_probe.py').read_text().replace('probe','batch'))
f=p/'.gitignore';f.write_text(f.read_text()+'/batch\n')
m=json.loads((p/'manifest.json').read_text());m['batch_sha256']=hashlib.sha256((p/'batch.cuh').read_bytes()).hexdigest();m['batch_mechanism']='Warp flatcombiner serializes published realissued packets in32request maxplus scan; no future reservation/no dedicatedCTA; same bandwidth and latency semantics; isolatednotproduction';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
