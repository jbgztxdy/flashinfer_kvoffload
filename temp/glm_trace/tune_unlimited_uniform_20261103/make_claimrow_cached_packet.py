from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'claimrow_quota.cuh').read_text()
a=' uint32_t count,epoch,done,current,waiting;';assert s.count(a)==1;s=s.replace(a,a+'\n uint32_t packet,packet_base,limit;')
a=''' const uint32_t n=ceil_div(s.service.count,20u),q=n/gridDim.y,r=n%gridDim.y,capacity=q+(blockIdx.y<r);
 if(ticket>=capacity)return n;
 return claimrow_packet_list(p)[blockIdx.y*q+min(blockIdx.y,r)+ticket];'''
b=''' if(ticket>=s.service.limit)return ceil_div(s.service.count,20u);
 return claimrow_packet_list(p)[s.service.packet_base+ticket];''';assert s.count(a)==1;s=s.replace(a,b)
a=''' uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,chunk=clean_service_chunk<KTraits>(p,s,ticket),b=ticket%1;
 if(t<20&&chunk*20+t<s.service.count){uint32_t slot=chunk*20+t;s.service.pages[b][t]=p.unique_pages[slot];s.service.victims[b][t]=p.victim_of[slot];}'''
b=''' uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,b=ticket%1;
 // Only warp0 queries this packet. Its leader alone writes packet; native
 // initialize/PV/step barriers publish it before any payload reader uses it.
 if(t<20){
  const uint32_t chunk=clean_service_chunk<KTraits>(p,s,ticket);
  if(t==0)s.service.packet=chunk;
  if(chunk*20+t<s.service.count){uint32_t slot=chunk*20+t;s.service.pages[b][t]=p.unique_pages[slot];s.service.victims[b][t]=p.victim_of[slot];}
 }''';assert s.count(a)==1;s=s.replace(a,b)
a='chunk=clean_service_chunk<KTraits>(p,s,ticket),b=ticket%1;';assert s.count(a)==2;s=s.replace(a,'chunk=s.service.packet,b=ticket%1;')
a=''' uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,ticket=s.service.current,chunk=clean_service_chunk<KTraits>(p,s,ticket);
 if(chunk*20>=s.service.count)return;
 if(t==0){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(clean_service_flags(p)+chunk),"r"(s.service.epoch+1):"memory");}'''
b=''' uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,ticket=s.service.current;
 if(ticket>=s.service.limit)return;
 // Other warps never read packet here. Only its writer (leader) reads the old
 // value, releases it, then writes the next value in ids: no cross-warp race.
 if(t==0){uint32_t chunk=s.service.packet;asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(clean_service_flags(p)+chunk),"r"(s.service.epoch+1):"memory");}''';assert s.count(a)==1;s=s.replace(a,b)
a=' if((clean_service_chunk<KTraits>(p,s,ticket))*20>=s.service.count)return;';assert s.count(a)==1;s=s.replace(a,' if(ticket>=s.service.limit)return;')
a='while((clean_service_chunk<KTraits>(p,s,s.service.done))*20<s.service.count)';assert s.count(a)==1;s=s.replace(a,'while(s.service.done<s.service.limit)')
a='if((clean_service_chunk<KTraits>(params,storage,storage.service.done))*20<storage.service.count)';assert s.count(a)==1;s=s.replace(a,'if(storage.service.done<storage.service.limit)')
a='s.service.produced=s.service.issues=s.service.wait_loops=s.service.rows=0;}';b='''s.service.produced=s.service.issues=s.service.wait_loops=s.service.rows=0;
  const uint32_t n=ceil_div(unique,20u),q=n/gridDim.y,r=n%gridDim.y;
  s.service.limit=q+(blockIdx.y<r);s.service.packet_base=blockIdx.y*q+min(blockIdx.y,r);s.service.packet=n;}''';assert s.count(a)==1;s=s.replace(a,b)
assert s.count('clean_service_chunk<KTraits>(p,s,ticket)')==1
assert not(p/'claimrow_cached_packet.cuh').exists();(p/'claimrow_cached_packet.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_claimrow_cached_packet');(p/'include_claimrow_cached_packet/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_claimrow_cached_packet.py').write_text((p/f'{n}_claimrow_quota.py').read_text().replace('claimrow_quota','claimrow_cached_packet'))
(p/'claimrow_cached_packet_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'claimrow_quota.cuh').read_bytes()).hexdigest(),mechanism='same exactbalanced packetplan; hoist perCTA packetbase/limit; onlywarp0query onceperpacket; leader-owned packet field reused forissue/stores/release; allpublication/reuse barriers retained'),indent=2)+'\n')
