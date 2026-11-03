"""Coalesce two far stages into one packet; keep the two attention stages."""
from pathlib import Path
import hashlib,json,shutil,difflib
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'packet.cuh').exists()
s=(p/'fused.cuh').read_text();a=s.index('// All256threads compute');b=s.index('template <bool SERVICE,typename KTraits,typename Params>',a);h=s[a:b]
h=h.replace('pages[2][10],victims[2][10]','pages[1][20],victims[1][20]').replace('remote[2][10*576]','remote[1][20*576]')
h=h.replace('10u','20u').replace('/10','/20').replace('*10','*20').replace('<10','<20').replace('ticket%2','ticket%1').replace('r<3','r<6').replace('v<720','v<1440').replace('ticket+2','ticket+1')
h=h.replace('fused_service_ids<KTraits>(p,s,0);fused_service_ids<KTraits>(p,s,1);','fused_service_ids<KTraits>(p,s,0);').replace('fused_service_issue<KTraits>(p,s,0);fused_service_issue<KTraits>(p,s,1);','fused_service_issue<KTraits>(p,s,0);')
s=s[:a]+h+s[b:]
# Same flaggranularity in launcher/kernelallocation andallstats readers.
s=s.replace('ceil_div(unique,10u)','ceil_div(unique,20u)').replace('const uint32_t fetch_rows=10;','const uint32_t fetch_rows=20;').replace('ceil_div(params.staging_capacity,10u)','ceil_div(params.staging_capacity,20u)')
s=s.replace('FusedService','PacketService').replace('fused_service_','packet_service_')
(p/'packet.cuh').write_text(s);shutil.copytree(root/'include',p/'include_packet');(p/'include_packet/flashinfer/attention/mla_host_batch.cuh').write_text(s)
base=(p/'base.cuh').read_text();(p/'packet.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production',tofile='packet',n=0)))
m=json.loads((p/'manifest.json').read_text());m['packet_sha256']=hashlib.sha256(s.encode()).hexdigest();m['packet_mechanism']='Merge2far10stages into1far20stage; same23040Bfarstorage,halfpacketfences,publishonce,readsoverlapPV; attention2stagesunchanged';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(p/'bench.py').read_text();s=s.replace('w.host_fetch_ctas=A.fetch_ctas','service_rows=20 if VARIANT==\'packet\' else 10\n    w.host_fetch_ctas=A.fetch_ctas')
s=s.replace("+9)//10", "+service_rows-1)//service_rows").replace("(miss+9)//10", "(miss+service_rows-1)//service_rows")
(p/'bench_packet.py').write_text(s)
s=(p/'test_fused.py').read_text().replace('+9)//10','+19)//20').replace("test_fused_output","test_packet_output");(p/'test_packet.py').write_text(s)
s=(p/'run_fused.py').read_text().split("m=json.loads((p/'manifest.json').read_text())")[0].replace("p/'run_state.json'","p/'packet_state.json'")
s+='''save()
run('correctness_packet','packet',['-m','pytest','-q','-s','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths',str(p/'test_packet.py')],True)
run('packet_native','packet',['-u',str(p/'bench_packet.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes','3','--settings','native_hbm','--fetch-ctas','24','--host-split','off'])
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
'''
(p/'run_packet.py').write_text(s)
