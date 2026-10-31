"""Summarize debug timestamps; these are diagnostic, not production benchmark results."""
import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
names=['start','prepare_done','prepare_barrier_done','resolve_done','before_fetch_barrier','fetch_start','fetch_done','attention_done','merge_barrier_done','merge_done','last_lru_start','last_lru_done']
records=[]
for path in sorted([*p.joinpath('raw').glob('profile_*.json'), *p.joinpath('raw').glob('profile_final_*.json')]):
 d=json.loads(path.read_text());flat=d['phase_timestamps_ns_flat'];ctas=[flat[i:i+16] for i in range(0,len(flat),16)];ctas=[x for x in ctas if x[0]>0]
 steps=[]
 for epoch in sorted({x[12] for x in ctas}):
  a=[x for x in ctas if x[12]==epoch];zero=min(x[0] for x in a)
  assert len(a)==a[0][13]==108
  assert all(x[14]==a[0][14] and x[15]==a[0][15] for x in a)
  phase={name:dict(min_us=(min(x[k] for x in a if x[k]>0)-zero)/1000,max_us=(max(x[k] for x in a if x[k]>0)-zero)/1000) for k,name in enumerate(names) if any(x[k]>0 for x in a)}
  steps.append(dict(epoch=epoch,phases=phase,full_rows=a[0][14],chunks=a[0][15]))
 assert len(steps)==128
 avg={name:dict(min_us=statistics.mean(x['phases'][name]['min_us'] for x in steps),max_us=statistics.mean(x['phases'][name]['max_us'] for x in steps)) for name in names}
 records.append(dict(raw=str(path),configuration=d['configuration'],means=avg,steps=steps))
(p/'phase_summary.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([r['means'] for r in records],indent=2))
