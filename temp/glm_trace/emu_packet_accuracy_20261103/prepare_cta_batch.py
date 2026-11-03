from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent
assert not(p/'batch_cta.cuh').exists()
s=(p/'batch.cuh').read_text();a=s.index('// Retain aggregate layout')
helper=r'''
// Explicit fullCTA hook, only for loaders where everythread participates after
// issuing its realcopies. All requests combined belong to THIS already-issued
// stage. This is not a lease for future stages/iterations.
template<uint32_t STAGES>
__device__ __forceinline__ void tile_cta(uint32_t stage,uint32_t far_rows,uint32_t row_bytes){
 if(!config.enabled)return;
 const uint32_t lane=threadIdx.x%32,t=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x,w=t/32;
 __shared__ uint32_t counts[8];
 const uint32_t sum=__reduce_add_sync(0xffffffffu,far_rows);
 if(lane==0)counts[w]=sum;
 __syncthreads();
 if(w==0){
  uint32_t local=lane<(blockDim.x*blockDim.y*blockDim.z)/32?counts[lane]:0;
  const uint32_t total=__reduce_add_sync(0xffffffffu,local);
  if(total){unsigned long long end;
   if(config.ps_per_byte)end=reserve_warp(static_cast<unsigned long long>(total)*row_bytes);
   else {end=now_ns()+config.latency_ns;end=broker_shfl(end,0);}
   if(lane==0)atomicMax(&pending_ready[cta_slot()][stage],end);
  }
 }
 __syncthreads();
}
'''
s=s[:a]+helper+s[a:];(p/'batch_cta.cuh').write_text(s);(p/'include/flashinfer/attention/far_link_emu.cuh').write_text(s)
# The frozen batch header already used for staged/uniform is kept unchanged.
# Direct is modified only after its first CPU compilation completes.
s=(p/'calibrate.cu').read_text();old='FAR_EMU_TILE(2,stage,ROWS==32?(threadIdx.x%32==0?max(0,min(4,count-chunk*ROWS-int(threadIdx.x/32)*4)):0):(threadIdx.x==0?min(ROWS,count-chunk*ROWS):0),1152);'
new='if constexpr(ROWS==32)flashinfer::far_emu::tile_cta<2>(stage,threadIdx.x%32==0?max(0,min(4,count-chunk*ROWS-int(threadIdx.x/32)*4)):0,1152);else FAR_EMU_TILE(2,stage,threadIdx.x==0?min(ROWS,count-chunk*ROWS):0,1152);'
assert old in s;s=s.replace(old,new)
s=s.replace('for(int config_mode:{0,1,2,3})','for(int config_mode:{2,3})')
(p/'calibrate_cta.cu').write_text(s);(p/'run_calibrate_cta.py').write_text((p/'run_probe.py').read_text().replace('probe','calibrate_cta'))
f=p/'.gitignore';f.write_text(f.read_text()+'/calibrate_cta\n')
m=json.loads((p/'manifest.json').read_text());m['batch_cta_sha256']=hashlib.sha256((p/'batch_cta.cuh').read_bytes()).hexdigest();(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
