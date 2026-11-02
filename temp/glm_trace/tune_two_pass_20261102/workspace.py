"""Reserve score workspace before CUDA events/Graph capture, identically for both versions."""
import inspect,textwrap
import flashinfer.mla._core as core
cls=core.BatchMLAPagedAttentionWrapper
source=textwrap.dedent(inspect.getsource(cls._run_host_batch))
needle='partials = 8 * get_device_sm_count(self.device)'
assert source.count(needle)==1
source=source.replace(needle,'partials = max(8 * get_device_sm_count(self.device), (rows * max_kv_len * 64 * 4 + 64 * 512 * 2 - 1) // (64 * 512 * 2))')
ns={};exec(source,core.__dict__,ns);cls._run_host_batch=ns['_run_host_batch']
