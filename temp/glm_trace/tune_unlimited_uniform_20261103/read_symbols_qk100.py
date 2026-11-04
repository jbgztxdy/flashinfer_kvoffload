"""Read/write registered CUDA symbols in private JIT library, outside all events.
ELF local host-shadow symbol addresses are given to cudaMemcpy{From,To}Symbol;
no device function or timed kernel is changed. Offsets verified from exact ELF.
"""
import ctypes as C,subprocess
from pathlib import Path
class Symbols:
 def __init__(self,variant):
  paths={z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if ('jit_'+variant+'/') in z and 'host_overlap_v2.so' in z}
  assert len(paths)==1,paths
  self.path=Path(next(iter(paths)))
  mappings=[z.split() for z in Path('/proc/self/maps').read_text().splitlines() if str(self.path) in z]
  self.base=min(int(z[0].split('-')[0],16)-int(z[2],16) for z in mappings)
  self.offsets={line.split()[-1]:int(line.split()[0],16) for line in subprocess.check_output(['nm','-a',str(self.path)],text=True).splitlines() if len(line.split())==3 and line.split()[0].strip('0123456789abcdefABCDEF')==''}
  runtime=next(z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in z)
  self.rt=C.CDLL(runtime);self.rt.cudaGetErrorString.restype=C.c_char_p
  self.rt.cudaMemcpyFromSymbol.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_int]
  self.rt.cudaMemcpyToSymbol.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_int]
 def addr(self,name):
  vals=[v for k,v in self.offsets.items() if k.endswith(name+'E')];assert len(vals)==1,(name,vals)
  return C.c_void_p(self.base+vals[0])
 def check(self,n):
  if n:raise RuntimeError(self.rt.cudaGetErrorString(n).decode())
 def get(self,name,ctype):
  value=ctype();self.check(self.rt.cudaMemcpyFromSymbol(C.byref(value),self.addr(name),C.sizeof(value),0,2));return value
 def put(self,name,value):self.check(self.rt.cudaMemcpyToSymbol(self.addr(name),C.byref(value),C.sizeof(value),0,1))
