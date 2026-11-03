from pathlib import Path
import json,statistics,hashlib
p=Path(__file__).resolve().parent;out=[]
for name in ['contiguous','rowlocal','reverse_packets','service_calls','waitvote','ticketstep','clean_ticketcontrol']:
 f=p/(name+'_native_unlimited_f24.json')
 if not f.exists():continue
 r=json.loads(f.read_text());means=[statistics.mean(a['us']) for a in r['passes']]
 if len(means)!=3:continue
 out.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),pass_means_us=means,last_vs_first_pct=100*(means[-1]/means[0]-1),last_slowest=means[-1]==max(means)))
result=dict(rows=out,last_slowest_count=sum(x['last_slowest'] for x in out),count=len(out),median_last_vs_first_pct=statistics.median(x['last_vs_first_pct'] for x in out),scope='existing normal-build runs; descriptive only; no cause established')
(p/'pass_drift_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
