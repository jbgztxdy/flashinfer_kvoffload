import torch,json,statistics,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;torch.set_num_threads(8)
a=p/'native_lru100_mapping.pt';r=torch.load(a,weights_only=False);out=[]
for t,d in enumerate(r['meta']):
 ids=d['indices'];n=(ids.reshape(-1,4)>=r['farstart']).sum(1);assert int(n.sum())==r['summaries'][t]['far_references']
 charges=((n*1152*10+999)//1000);b=int(n.sum())*1152
 out.append(dict(step=t,far_bytes=b,packets=int((n>0).sum()),budget_ns=b/100,charged_integer_ns=int(charges.sum()),rounding_ns=int(charges.sum())-b/100))
x=dict(kind='CPU source-model accounting; not runtime hardware counters',mapping_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),steps=out,mean_far_MB=statistics.mean(d['far_bytes'] for d in out)/1e6,mean_packets=statistics.mean(d['packets'] for d in out),mean_rounding_us=statistics.mean(d['rounding_ns'] for d in out)/1000)
(p/'native_lru100_packet_audit.json').write_text(json.dumps(x,indent=2));print({k:v for k,v in x.items() if k!='steps'})
