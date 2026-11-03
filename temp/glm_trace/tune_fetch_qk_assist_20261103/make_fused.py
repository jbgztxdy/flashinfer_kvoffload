"""One fused-pair follow-up, no parameter sweep or production edit."""
from pathlib import Path
import difflib,hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'fused.cuh').exists()
s=(p/'candidate.cuh').read_text();h=(p/'assist_helpers.cuh').read_text();f=h
f=f.replace('unsigned long long job[kAssistPairs];','unsigned long long job[kAssistPairs];\n  unsigned long long worker_wait_ns,fetch_begin,fetch_end,helper_end;')
f=f.replace('void assist_qk32(', 'void assist_qk_pair(')
f=f.replace('T* qs=work;T* ks=work+4096;float s[2][8];','T* qs=work;T* ks=work+4096;float s[4][8];')
f=f.replace('for(uint32_t v=t;v<256;v+=128){uint32_t key=tile*32+v/8;', 'for(uint32_t v=t;v<512;v+=128){uint32_t r=v/8;uint32_t key=(tile-r/32)*32+r%32;')
f=f.replace('for(uint32_t g=0;g<2;++g)', 'for(uint32_t g=0;g<4;++g)')
f=f.replace('out[(g*128+t)*8+j]=s[g][j];', 'out[(g/2)*2048+((g%2)*128+t)*8+j]=s[g][j];')
f=f.replace('bool waited=false;', 'bool waited=false;unsigned long long started=0;')
f=f.replace('if(state==1){waited=true;__nanosleep(64);continue;}', 'if(state==1){if(!waited)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(started));waited=true;__nanosleep(64);continue;}')
f=f.replace('c->waited+=waited;return;', 'c->waited+=waited;if(waited){unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));c->worker_wait_ns+=now-started;}return;')
old='''  assist_qk32(p,job_row,job_base,job_len,hi,output,reinterpret_cast<typename Params::DTypeKV*>(smem+147968));
  assist_qk32(p,job_row,job_base,job_len,hi-1,output+2048,reinterpret_cast<typename Params::DTypeKV*>(smem+147968));'''
new='''  assist_qk_pair(p,job_row,job_base,job_len,hi,output,reinterpret_cast<typename Params::DTypeKV*>(smem+147968));'''
assert old in f;f=f.replace(old,new)
f=f.replace('if(t==0)atomicAdd(&all->helper_done,1u);', 'if(t==0){unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));all[blockIdx.y].helper_end=now;atomicAdd(&all->helper_done,1u);}')
assert s.count(h)==1;s=s.replace(h,f).replace('FetchAssist','FusedFetchAssist').replace('fetch_assist_','fused_fetch_assist_')
s=s.replace('service_work[12288]','service_work[16384]').replace('static_assert(fused_fetch_assist_smem == 160256);','static_assert(fused_fetch_assist_smem == 164352);')
s=s.replace('c->produced=c->consumed=c->canceled=c->waited=0;', 'c->produced=c->consumed=c->canceled=c->waited=0;c->worker_wait_ns=c->fetch_begin=c->fetch_end=c->helper_end=0;')
s=s.replace('if(tid==0)producer_finished=0;', 'if(tid==0){producer_finished=0;unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));assist_controls(params)[blockIdx.y].fetch_begin=now;}')
s=s.replace('if(tid==0){atomicAdd(&assist_controls(params)->fetch_done,1u);', 'if(tid==0){unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));assist_controls(params)[blockIdx.y].fetch_end=now;atomicAdd(&assist_controls(params)->fetch_done,1u);')
(p/'fused_helpers.cuh').write_text(f);(p/'fused.cuh').write_text(s);shutil.copytree(root/'include',p/'include_fused');(p/'include_fused/flashinfer/attention/mla_host_batch.cuh').write_text(s)
base=(p/'include_base/flashinfer/attention/mla_host_batch.cuh').read_text();(p/'fused.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production',tofile='fused',n=0)))
m=json.loads((p/'manifest.json').read_text());m['fused_sha256']=hashlib.sha256(s.encode()).hexdigest();m['fused_change']='Q shared load once per64KV pair,18instead36groupbarriers,one16KiB featureworkspace; diagnostic wait/fetch times';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
