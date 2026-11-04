import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
root=p.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
names=['service_poll_g100_l0_f24.json','service_poll_g100_l20000_f24.json',
       'poll_control_next_packet_qk_next100_issuebar_g100_l20000_f24.json',
       'poll_reverse_service_poll_g100_l20000_f24.json',
       'poll_reverse_next_packet_qk_next100_issuebar_g100_l20000_f24.json']
out=[]
for name in names:
    j=json.loads((p/name).read_text())
    assert j['complete'] and j['cache_payload_slots']==524288
    assert len(j['passes'])==1 and len(j['passes'][0]['us'])==32
    for lib in j['libraries']:assert sha(root/lib['path'])==lib['sha256']
    if j['variant']=='service_poll':assert j['header_sha256']==sha(p/'service_poll.cuh')
    out.append(dict(raw=name,raw_sha256=sha(p/name),mean_us=sum(j['passes'][0]['us'])/32,
                    mean_unique_pages=sum(j['passes'][0]['unique_misses'])/32,
                    libraries=j['libraries']))
a=json.loads((p/'accounting_service_poll.json').read_text())
assert a['complete'] and a['header_sha256']==sha(p/'service_poll.cuh')
assert sha(a['symbol_library'])==a['library_sha256']
for case in a['cases']:
    assert case['recurrence_exact'] and case['actual_far_rows']==case['expected_unique_rows']
    assert case['request_count']==(case['actual_far_rows']+19)//20
    assert case['actual_far_bytes']==case['actual_far_rows']*1152
(p/'service_poll_audit.json').write_text(json.dumps(dict(complete=True,raw=out,accounting=a['cases']),indent=2)+'\n')
print('5 complete timings and loaded library SHA; exact unique accounting verified')
