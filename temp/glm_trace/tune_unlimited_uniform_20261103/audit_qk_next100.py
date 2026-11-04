"""Audit isolated QK-ahead service timing and actual loaded source/library bytes."""
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name, header, setting):
    raw = p / (name + '_' + setting + '_f24.json')
    j = json.loads(raw.read_text())
    assert j['complete'] and j['config']['emulator'] == 'batch_call_issuebar'
    assert j['config']['staged'] and j['config']['steps'] == 32
    assert j['cache_payload_slots'] == 524288 and len(j['passes']) == 1
    assert j['header_sha256'] == digest(p / header)
    assert j['far_sha256'] == digest(p / 'include_next_packet_qk_next100_issuebar/flashinfer/attention/far_link_emu.cuh')
    for lib in j['libraries']:
        assert digest(root / lib['path']) == lib['sha256'], lib
    step = j['passes'][0]
    assert len(step['us']) == len(step['unique_misses']) == 32
    return dict(raw=raw.name, raw_sha256=digest(raw), header_sha256=j['header_sha256'],
                loaded_host_sha256=next(x['sha256'] for x in j['libraries'] if 'host_overlap_v2' in x['path']),
                mean_us=sum(step['us']) / 32, mean_unique_pages=sum(step['unique_misses']) / 32,
                us=step['us'])


rows = {}
for setting in ('g100_l0', 'g100_l20000'):
    early = read('next_packet_qk_next100_issuebar', 'next_packet_qk_issuebar.cuh', setting)
    clean = read('clean_next100_issuebar', 'clean_issuebar.cuh', setting)
    late = read('next_packet_late_issuebar', 'next_packet_late_issuebar.cuh', setting)
    rows[setting] = dict(early=early, clean=clean, late=late,
                         vs_clean_us=clean['mean_us']-early['mean_us'],
                         vs_late_us=late['mean_us']-early['mean_us'],
                         vs_late_percent=100*(late['mean_us']-early['mean_us'])/late['mean_us'],
                         steps_faster_than_clean=sum(a < b for a,b in zip(early['us'],clean['us'])),
                         steps_faster_than_late=sum(a < b for a,b in zip(early['us'],late['us'])))

account = json.loads((p/'accounting_next_packet_qk_next100_issuebar.json').read_text())
assert account['complete'] and account['header_sha256'] == digest(p/'next_packet_qk_issuebar.cuh')
assert digest(account['symbol_library']) == account['library_sha256']
for case in account['cases']:
    assert case['actual_far_rows'] == case['expected_unique_rows']
    assert case['request_count'] == (case['actual_far_rows']+19)//20
    assert case['actual_far_bytes'] == case['actual_far_rows']*1152
    assert case['recurrence_exact']

normal = json.loads((p/'next_packet_qk_native_unlimited_f24.json').read_text())
assert normal['complete'] and normal['header_sha256'] == digest(p/'next_packet_qk.cuh')
assert normal['cache_payload_slots'] == 524288 and len(normal['passes']) == 3
for lib in normal['libraries']:
    assert digest(root / lib['path']) == lib['sha256']

out = dict(complete=True, strict_settings=rows, accounting_cases=account['cases'],
           accounting_library_sha256=account['library_sha256'],
           normal_pass_means_us=[sum(x['us'])/32 for x in normal['passes']],
           normal_header_sha256=normal['header_sha256'],
           production_header_sha256=digest(root/'include/flashinfer/attention/mla_host_batch.cuh'))
(p/'qk_next100_audit.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:{x:v for x,v in t.items() if x not in ('early','late','clean')}
                  for k,t in rows.items()}, indent=2))
