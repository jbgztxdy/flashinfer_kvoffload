import ctypes
from pathlib import Path
def inspect_graph(g):
    libpath=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line)
    rt=ctypes.CDLL(libpath)
    rt.cudaGetErrorString.restype=ctypes.c_char_p
    rt.cudaFuncGetName.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.c_void_p]
    def check(code):
        if code:raise RuntimeError(rt.cudaGetErrorString(code).decode())
    class Dim3(ctypes.Structure):_fields_=[('x',ctypes.c_uint),('y',ctypes.c_uint),('z',ctypes.c_uint)]
    class KernelParams(ctypes.Structure):
        _fields_=[('func',ctypes.c_void_p),('grid',Dim3),('block',Dim3),('shared',ctypes.c_uint),('params',ctypes.c_void_p),('extra',ctypes.c_void_p)]
    rt.cudaGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
    rt.cudaGraphNodeGetType.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    rt.cudaGraphKernelNodeGetParams.argtypes=[ctypes.c_void_p,ctypes.POINTER(KernelParams)]
    rt.cudaOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
    graph=ctypes.c_void_p(g.raw_cuda_graph());count=ctypes.c_size_t()
    check(rt.cudaGraphGetNodes(graph,None,ctypes.byref(count)))
    nodes=(ctypes.c_void_p*count.value)();check(rt.cudaGraphGetNodes(graph,nodes,ctypes.byref(count)))
    launches=[]
    for node in nodes:
        kind=ctypes.c_int();check(rt.cudaGraphNodeGetType(node,ctypes.byref(kind)))
        if kind.value!=0:continue
        cfg=KernelParams();check(rt.cudaGraphKernelNodeGetParams(node,ctypes.byref(cfg)))
        name=ctypes.c_char_p();check(rt.cudaFuncGetName(ctypes.byref(name),cfg.func))
        threads=cfg.block.x*cfg.block.y*cfg.block.z;blocks=ctypes.c_int()
        check(rt.cudaOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks),cfg.func,threads,cfg.shared))
        launches.append(dict(grid=[cfg.grid.x,cfg.grid.y,cfg.grid.z],block=[cfg.block.x,cfg.block.y,cfg.block.z],
            dynamic_shared_bytes=cfg.shared,active_blocks_per_SM=blocks.value,kernel_name=name.value.decode()))
    return launches
