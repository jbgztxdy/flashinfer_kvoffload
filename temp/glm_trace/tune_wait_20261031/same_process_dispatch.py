"""Verify which CUDA functions the earlier same-allocation control actually captures."""
import ctypes as C
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
original=p/'same_process_control.py'
prefix,tail=original.read_text().split('\nrecords=[]\n',1)
prefix=prefix.replace('torch.cuda.CUDAGraph()','torch.cuda.CUDAGraph(keep_graph=True)')
ns={'__file__':str(original),'__name__':'same_process_dispatch_bootstrap'}
exec(compile(prefix,str(original),'exec'),ns)
path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line)
rt=C.CDLL(path);rt.cudaGetErrorString.restype=C.c_char_p
def check(code):
    if code:raise RuntimeError(rt.cudaGetErrorString(code).decode())
class Dim3(C.Structure):_fields_=[('x',C.c_uint),('y',C.c_uint),('z',C.c_uint)]
class KernelParams(C.Structure):
    _fields_=[('func',C.c_void_p),('grid',Dim3),('block',Dim3),('shared',C.c_uint),('params',C.c_void_p),('extra',C.c_void_p)]
rt.cudaGraphGetNodes.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
rt.cudaGraphNodeGetType.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
rt.cudaGraphKernelNodeGetParams.argtypes=[C.c_void_p,C.POINTER(KernelParams)]
dispatch={}
for variant,graph in ns['graphs'].items():
    raw=C.c_void_p(graph.raw_cuda_graph());count=C.c_size_t()
    check(rt.cudaGraphGetNodes(raw,None,C.byref(count)))
    nodes=(C.c_void_p*count.value)();check(rt.cudaGraphGetNodes(raw,nodes,C.byref(count)))
    selected=[]
    for node in nodes:
        kind=C.c_int();check(rt.cudaGraphNodeGetType(node,C.byref(kind)))
        if kind.value!=0:continue
        cfg=KernelParams();check(rt.cudaGraphKernelNodeGetParams(node,C.byref(cfg)))
        if (cfg.block.x,cfg.block.y,cfg.block.z)!=(32,4,2):continue
        selected.append(dict(function=hex(cfg.func),grid=[cfg.grid.x,cfg.grid.y,cfg.grid.z],
                             block=[32,4,2],dynamic_shared_bytes=cfg.shared))
    assert len(selected)==1024 and all(x==selected[0] for x in selected)
    dispatch[variant]=dict(total_graph_nodes=count.value,attention_nodes=len(selected),launch=selected[0])
record=dict(complete=False,original_control_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dispatch=dispatch,
    resources_start=ns['ns']['RESOURCE_START'],resources_end=ns['ns']['require_exclusive']())
out=p/'same_process_dispatch.json';out.write_text(json.dumps(record,indent=2)+'\n')
assert len({r['launch']['function'] for r in dispatch.values()})==2,'Both wrappers captured the same function'
record['complete']=True;out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
