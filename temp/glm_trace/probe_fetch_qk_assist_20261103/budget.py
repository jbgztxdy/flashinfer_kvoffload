"""Read-only feasibility accounting. No unmeasured kernel benefit is inferred."""
from pathlib import Path
import hashlib,json,statistics as S
p=Path(__file__).resolve().parent;root=p.parents[2]
d= json.loads((p.parent/'diagnose_fetch_sm_v2_20261103/analysis.json').read_text())
coords=[]
for thread in range(256):
 lane=thread%32;warp=thread//32;head=(warp%4)*16+lane//4;col=(lane%4)*2
 for i in range(8):coords.append((head+((i%4)//2)*8,(warp//4)*16+col+(i%2)+(i//4)*8))
assert len(coords)==len(set(coords))==64*32 and set(coords)=={(h,k) for h in range(64) for k in range(32)}
for w in range(8):
 aa=[];bb=[]
 for lane in range(32):
  h=(w%4)*16+lane//4;c=(lane%4)*2;k=(w//4)*16+lane//4
  aa += [(h,c+x) for x in [0,1,8,9]]+[(h+8,c+x) for x in [0,1,8,9]]
  bb += [(k,c+x) for x in [0,1,8,9]]+[(k+8,c+x) for x in [0,1,8,9]]
 assert len(set(aa))==len(set(bb))==256
features=[(512+dim*16 if dim<4 else (dim-4)*16)+x for dim in range(36) for x in range(16)]
assert features==list(range(512,576))+list(range(512))
rows=[]
for tr in d['trace']:
 raw=json.loads((p.parent/'diagnose_fetch_sm_v2_20261103'/('trace_'+tr['setting']+'.json')).read_text())
 chunks=S.mean(sum(z[29] for z in epoch['ctas']) for epoch in raw['passes'][0]['timeline'])
 m=tr['means'];rows.append(dict(setting=tr['setting'],chunks=chunks,per_fetchCTA_chunks=chunks/24,
 average_us_per_fetch_chunk=m['fetch_sm_us']/chunks,far_wait_us_per_chunk=m['far_wait_per_fetch_cta_us']*24/chunks,
 physical_wait_us_per_chunk=m['physical_wait_per_fetch_cta_us']*24/chunks,
 explicit_wait_service_equivalent_wall_us=(m['far_wait_per_fetch_cta_us']+m['physical_wait_per_fetch_cta_us'])*24/108,
 explicit_wait_limit='Opportunity estimate from intrusive timer; includes barrier effects, not causal savings or hard bound. It cannot establish idle tensor-core capacity during memory work.'))
r=dict(classification='CPU coordinate coverage and measured-budget analysis; GPU probe not executed',
 production_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest(),
 far_sha256=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest(),
 probe_source_sha256=hashlib.sha256((p/'probe.cu').read_bytes()).hexdigest(),
 frozen_binary_sha256=hashlib.sha256((p/'probe').read_bytes()).hexdigest(),
 cpu_mapping=dict(output_coordinates=len(coords),unique_output_coordinates=len(set(coords)),features=len(features),operand_coverage_per_warp=256,pe_then_ckv=True),
 rows=rows,
 paired_score_bytes=64*64*4,bounded_mailbox_bytes_84CTAs_2pairs=84*2*64*64*4,
 fetch_buffer_bytes=2*64*576*2,original_dynamic_shared=147968,original_static_shared=2304,
 missing_evidence=['GPU QK arithmetic/payload check','probe link100Gb throughput with and without QK','actual mailbox integration and work replacement','84worker interference','full kernel codegen/correctness/performance'])
(p/'budget.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
