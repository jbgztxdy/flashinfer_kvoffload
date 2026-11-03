from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent
shutil.copy2(p/'batch.cuh',p/'batch_first.cuh')
s=(p/'batch.cuh').read_text().replace('__device__ __forceinline__ void broker_combine(){','__device__ __forceinline__ void broker_combine(){\n asm volatile("fence.acq_rel.gpu;":::"memory");')
(p/'batch.cuh').write_text(s);(p/'include/flashinfer/attention/far_link_emu.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['batch_first_sha256']=hashlib.sha256((p/'batch_first.cuh').read_bytes()).hexdigest();m['batch_sha256']=hashlib.sha256((p/'batch.cuh').read_bytes()).hexdigest();m['batch_first_result_header']='batch_first.cuh';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(p/'batch.cu').read_text();s=s.replace('for(int config_mode:{0,1,2})','for(int config_mode:{2})').replace('for(int rep=0;rep<4;++rep)','for(int rep=0;rep<2;++rep)')
s=s.replace('FAR_EMU_INIT(2);FAR_EMU_SYNC();','FAR_EMU_INIT(2);FAR_EMU_SYNC();__shared__ U rec[2];')
s=s.replace('FAR_EMU_TILE(2,stage,threadIdx.x==0?min(ROWS,count-chunk*ROWS):0,1152);','FAR_EMU_TILE(2,stage,threadIdx.x==0?min(ROWS,count-chunk*ROWS):0,1152);if(threadIdx.x==0)rec[stage]=flashinfer::far_emu::broker_requests[blockIdx.x*8].record;')
s=s.replace('FAR_EMU_WAIT(2,step%2);__syncthreads();','FAR_EMU_WAIT(2,step%2);__syncthreads();if(threadIdx.x==0)flashinfer::far_emu::broker_records[rec[step%2]].consume=flashinfer::far_emu::now_ns();')
s=s.replace('int main(){ck(cudaSetDevice(0));','int main(){ck(cudaSetDevice(0));unsigned tracing=1;ck(cudaMemcpyToSymbol(flashinfer::far_emu::broker_trace,&tracing,sizeof(tracing)));')
needle='if(rep){if(!first)'
code='''U entries;ck(cudaMemcpyFromSymbol(&entries,flashinfer::far_emu::broker_count,sizeof(entries)));std::vector<flashinfer::far_emu::BrokerRecord> records(entries);ck(cudaMemcpyFromSymbol(records.data(),flashinfer::far_emu::broker_records,entries*sizeof(records[0])));U clock=0,charged=0;int bad=0,early=0;for(auto& x:records){U expected=std::max(clock,x.issue)+x.cost;if(x.end!=expected||x.prior!=expected-x.cost)++bad;if(x.consume<x.end+lat)++early;clock=x.end;charged+=x.cost;}U expectedCost=0;for(int row=0;row<count;row+=(mode?10:64))expectedCost+=(U(std::min(mode?10:64,count-row))*1152*10+999)/1000;if(charged!=expectedCost||entries!=(count+(mode?9:63))/(mode?10:64))++bad;if(bad||early){fprintf(stderr,"auditbad=%d early=%d\\n",bad,early);return 4;}'''
assert needle in s;s=s.replace(needle,code+needle)
s=s.replace('\\\"us\\\":%.6f}', '\\\"us\\\":%.6f,\\\"request_records\\\":%llu,\\\"charged_ns\\\":%llu,\\\"bad\\\":%d,\\\"early\\\":%d}')
s=s.replace('rep,ms*1000);','rep,ms*1000,entries,charged,bad,early);')
(p/'batch_verify.cu').write_text(s)
s=(p/'run_probe.py').read_text().replace('probe','batch_verify');(p/'run_batch_verify.py').write_text(s)
f=p/'.gitignore';f.write_text(f.read_text()+'/batch_verify\n')
