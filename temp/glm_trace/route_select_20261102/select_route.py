"""Apply an offline measured route once, before warming/capturing a wrapper.

Profiles are restricted to this source/hardware/workload. There is no online timing,
per-launch winner selection, hidden fallback to another mathematical/cache policy,
or claim that unmeasured shapes have a known optimal route.
"""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
HARDWARE='A100-SXM4-40GB NV12 NUMA0'
def select_route(*, hardware, requests, mtp, capacity, setting, fraction=None, topk=2048, num_heads=64, dtype="bfloat16"):
    if (topk,num_heads,dtype)!=(2048,64,"bfloat16"):
        raise ValueError("No measured profile for this attention shape or dtype")
    if hardware!=HARDWARE:
        raise ValueError('No measured route profile for this hardware')
    if capacity not in (0,4096):
        raise ValueError('No measured profile for this cache capacity')
    kind='lru' if capacity else 'nocache'
    key=(kind,requests,mtp,None if capacity else fraction,setting)
    records=json.loads((P/'selected_routes.json').read_text())['profiles']
    for row in records:
        actual=tuple(row[k] for k in ['kind','requests','mtp','fraction','setting'])
        if actual==key:
            return dict(route=row['selected_route'],host_stage=row['selected_route']=='staged',profile=row)
    raise ValueError('No matched staged/inplace comparison for this workload')
def configure_wrapper(wrapper, **workload):
    if getattr(wrapper,'_host_batch_buffers',None) is not None:
        raise ValueError('Choose a route before wrapper warmup and CUDA Graph capture')
    choice=select_route(**workload)
    wrapper.host_stage=choice['host_stage']
    wrapper.host_fetch_ctas=24 if workload['requests']*workload['mtp']>=432 else 16
    wrapper.host_split=workload['requests']*workload['mtp']<432
    return choice
