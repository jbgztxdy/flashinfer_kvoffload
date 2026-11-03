from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent
s=(p/'clean.cuh').read_text()
helper='''__device__ __forceinline__ uint32_t reverse_packet_chunk(uint32_t count,uint32_t ticket){
 const uint32_t total=ceil_div(count,20u);
 const uint32_t jobs=blockIdx.y<total?ceil_div(total-blockIdx.y,uint32_t(gridDim.y)):0;
 return ticket<jobs?blockIdx.y+(jobs-1-ticket)*gridDim.y:total;
}
'''
s=s.replace('struct CleanService {',helper+'struct CleanService {')
for a,b in [('blockIdx.y+ticket*gridDim.y','reverse_packet_chunk(s.service.count,ticket)'),('blockIdx.y+s.service.done*gridDim.y','reverse_packet_chunk(s.service.count,s.service.done)'),('blockIdx.y+storage.service.done*gridDim.y','reverse_packet_chunk(storage.service.count,storage.service.done)')]:
 assert a in s;s=s.replace(a,b)
assert not (p/'reverse_packets.cuh').exists()
(p/'reverse_packets.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_reverse_packets');(p/'include_reverse_packets/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_reverse_packets.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','reverse_packets'))
f=p/'bench_reverse_packets.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
cases=0
for total in [0,1,19,20,21,107,108,109,1000,21528]:
 for grid in [1,2,24,108]:
  chunks=[]
  for c in range(grid):
   forward=list(range(c,total,grid));back=list(reversed(forward));chunks+=back
   assert set(back)==set(forward)
  assert sorted(chunks)==list(range(total));cases+=1
(p/'reverse_packets_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base_sha256':hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),'partition_cases':cases,'mechanism':'same round-robin ownership, reverse per-owner packet sequence only'},indent=2))
