"""Correct the temporary owner encoding without changing a queued build's inputs."""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
s = (p / 'owner_cache.cuh').read_text()
assert hashlib.sha256(s.encode()).hexdigest() == '39a2e31bd0b5b95651e4e0d8367d46e34ed2fa6cf3cf9cf67a307a4c34a7e1f0'
replacements = [
    ('if (host_batch_caching(params) && (fresh_masks[i] >> lane & 1)) entry |= kHostBatchOwnerBit;',
     'if (host_batch_caching(params) && (fresh_masks[i] >> lane & 1)) entry = ~entry;'),
    ('const IdType page = Params::whole_rows ? marked & ~kHostBatchOwnerBit : marked;',
     'const IdType page = Params::whole_rows && marked < 0 ? ~marked : marked;'),
    ('if (marked & kHostBatchOwnerBit) entry &= ~kHostBatchOwnerBit;',
     'if (marked < 0) entry &= ~kHostBatchOwnerBit;')]
for old, new in replacements:
    assert s.count(old) == 1
    s = s.replace(old, new)
v = 'owner_complement'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256((p / 'base.cuh').read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Preserve prepare CAS owner by complementing its temporary nonnegative Host ID. Resolve restores full ID and clears bit30 of bounded negative staging entry only for that owner. Attention fills cache from shared KV tile without atomic claims; fetch only stages/publishes.'), indent=2) + '\n')
print((p / ('manifest_' + v + '.json')).read_text())
