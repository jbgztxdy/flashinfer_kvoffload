"""Use CTAbarrier->GPUrelease->GPUacquire transitivity for packet publication."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'release.cuh').exists()
s=(p/'packet.cuh').read_text();a=s.index('void packet_service_stores');b=s.index('// Called immediately after',a);t=s[a:b];assert t.count('__threadfence();')==1;t=t.replace('__threadfence();','// The following existing jointMDO CTA barrier plus leader st.release.gpu\n // publishes all writers transitively; consumers use ld.acquire.gpu.');s=s[:a]+t+s[b:]
s=s.replace('PacketService','ReleaseService').replace('packet_service_','release_service_');(p/'release.cuh').write_text(s);shutil.copytree(root/'include',p/'include_release');(p/'include_release/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['release_sha256']=hashlib.sha256(s.encode()).hexdigest();m['release_protocol_sources']=['https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model-release-acquire-patterns','https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model-causality-order','https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar'];(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(p/'bench_packet.py').read_text().replace("VARIANT=='packet'", "VARIANT in ['packet','release']");(p/'bench_release.py').write_text(s)
(p/'test_release.py').write_text((p/'test_packet.py').read_text().replace('test_packet_output','test_release_output'))
s=(p/'run_packet.py').read_text().replace("p/'packet_state.json'","p/'release_state.json'").replace("'correctness_packet','packet'","'correctness_release','release'").replace("p/'test_packet.py'","p/'test_release.py'").replace("'packet_native','packet'","'release_native','release'").replace("p/'bench_packet.py'","p/'bench_release.py'");(p/'run_release.py').write_text(s)
