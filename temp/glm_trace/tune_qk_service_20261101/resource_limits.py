"""Read driver device limits without creating a context or launching GPU work."""
import ctypes as C,json,time
from pathlib import Path
p=Path(__file__).resolve().parent
lib=C.CDLL('libcuda.so.1')
def check(x):
 if x:raise RuntimeError('CUDA driver error '+str(x))
check(lib.cuInit(0));dev=C.c_int();check(lib.cuDeviceGet(C.byref(dev),0))
name=C.create_string_buffer(128);check(lib.cuDeviceGetName(name,128,dev))
attrs={}
for label,number in [('max_registers_per_block',12),('max_shared_per_sm',81),('max_registers_per_sm',82),('max_shared_per_block_optin',97)]:
 value=C.c_int();check(lib.cuDeviceGetAttribute(C.byref(value),number,dev));attrs[label]=value.value
record=dict(timestamp=time.strftime('%FT%T%z'),physical_gpu=3,name=name.value.decode(),device_attributes=attrs,
 no_context_created=True,no_gpu_work_launched=True,
 packing_review=dict(current_compute=140032,current_measured_static=1408,current_two8row_shared=18432,
    largest_three_equal_shared_rows_if_static_same=(attrs['max_shared_per_block_optin']-140032-1408)//(3*1152),
    warning='Packing arithmetic is not a compiled-resource,dispatch,correctness or performance proof. Any future implementation must query actual kernel shared attributes and max-block limit before setting dynamic allocation.'))
(p/'resource_limits.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
