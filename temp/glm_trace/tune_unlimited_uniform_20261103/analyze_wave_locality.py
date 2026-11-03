from pathlib import Path
import json,bisect,hashlib
p=Path(__file__).resolve().parent
src=p/'claimrow_timeline_packet_weights_native_unlimited_f24.json'
r=json.loads(src.read_text());assert r['complete']
bases=r['packet_weight_inputs']['row_slot_base'];n=(bases[-1]+19)//20;grid=108
preferred=[bisect.bisect_right(bases[1:],packet*20)%grid for packet in range(n)]
local_by_wave=[len(set(preferred[at:at+grid])) for at in range(0,n,grid)]
summary=dict(input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='actual final step only; upper bound if each CTA takes exactly one packet per original 108-packet wave, noGPU reassignment or timing',packets=n,waves=len(local_by_wave),max_local_packets=sum(local_by_wave),max_local_fraction=sum(local_by_wave)/n,mean_distinct_owner_per_wave=sum(local_by_wave)/len(local_by_wave),per_wave=local_by_wave)
(p/'claimrow_wave_locality.json').write_text(json.dumps(summary,indent=2)+chr(10))
print({k:v for k,v in summary.items() if k not in ['per_wave','input_sha256']})
