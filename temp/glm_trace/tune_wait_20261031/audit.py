"""Reconstruct CTA budgets; diagnostic timers are not additive phase overheads."""
import hashlib
import json
import math
import statistics
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
records = []
for f in sorted((p / 'raw').glob('*.json')):
    d = json.loads(f.read_text())
    if not d.get('complete'):
        continue
    cfg = d['configuration']
    if cfg['variant'] != 'profile_wait':
        continue
    assert cfg['mtp'] == 8 and cfg['positions'] == 1024 and cfg['passes'] == 1
    assert cfg['requests'] == 128 and cfg['fetch_ctas'] == 24 and not cfg['host_split']
    assert d['resources_start']['foreign'] == d['resources_end']['foreign'] == []
    for name, h in d['source_sha256'].items():
        assert sha(p / 'source_snapshots' / (h + Path(name).suffix)) == h
    c = d['cases'][0]
    assert max(c['numerical_checks']['max_abs_error'].values()) < .01
    assert all(x == c['far_unique'][0] for x in c['ours_unique_miss_samples'][0])
    for mode, t in c['timings'].items():
        samples = t['per_step_samples_us']
        assert len(samples) == 128 and all(len(x) == 1 for x in samples)
        assert abs(statistics.mean(x[0] for x in samples) - t['mean_us']) < 1e-7
        assert all(math.isfinite(x[0]) and x[0] > 0 for x in samples)
        end = 0
        for guard in [g for g in c['guards'] if g['mode'] == mode and g['pass_index'] == 0]:
            assert guard['valid'] and guard['first'] == end
            if guard['timing_queue'] != 'cuda_graph':
                assert guard['guard_ms'] >= guard['enqueue_ms'] + max(1, .05 * guard['enqueue_ms'])
            end = guard['last']
        assert end == 128
    flat = d['wait_diagnostic_flat']
    assert len(flat) == 128 * 128 * 32
    steps = []
    for slot in range(128):
        active = [(cta, flat[(slot * 128 + cta) * 32:(slot * 128 + cta + 1) * 32]) for cta in range(128)]
        active = [(cta, x) for cta, x in active if x[0]]
        assert len(active) == 108
        epoch = active[0][1][12]
        assert all(x[12] == epoch and x[13:16] == [108, 1024, 1] for _, x in active)
        assert epoch % 128 == slot
        assert sum(x[20] for _, x in active) == 1024
        assert sum(x[22] for _, x in active) == 128
        assert all(all(x[i] <= x[i+1] for i in range(8)) for _, x in active)
        assert all(x[7] <= x[26] <= x[8] for _, x in active if x[20])
        assert all(x[18] <= x[17] and x[16] <= x[19] for _, x in active)
        origin = min(x[0] for _, x in active)
        row = dict(epoch=epoch,
            fetch_start_us=(max(x[6] for _, x in active)-origin)/1000,
            fetch_end_us=(max(x[7] for _, x in active)-origin)/1000,
            attention_end_us=(max(x[8] for _, x in active)-origin)/1000,
            wait_us_per_attention_item=sum(x[16] for _, x in active) / 1000 / 1024,
            attention_us_per_item=sum(x[19] for _, x in active) / 1000 / 1024,
            blocked_call_fraction=sum(x[18] for _, x in active) / sum(x[17] for _, x in active),
            ctas=[])
        for cta, x in active:
            row['ctas'].append(dict(cta=cta, fetcher=cta>=84, wait_us=x[16]/1000,
                wait_calls=x[17], blocked_calls=x[18], attention_us=x[19]/1000,
                attention_items=x[20], lru_us=x[21]/1000, lru_items=x[22],
                before_items_us=(x[7]-origin)/1000, done_us=(x[8]-origin)/1000,
                last_attention_start_us=(x[26]-origin)/1000 if x[26] else None))
        steps.append(row)
    summaries = {}
    for group, select in [('all', lambda x: True), ('workers', lambda x: not x['fetcher']), ('fetchers', lambda x: x['fetcher'])]:
        ctas = [x for s in steps for x in s['ctas'] if select(x)]
        wait = sum(x['wait_us'] for x in ctas)
        attention = sum(x['attention_us'] for x in ctas)
        summaries[group] = dict(mean_wait_us_per_CTA=statistics.mean(x['wait_us'] for x in ctas),
            mean_attention_us_per_CTA=statistics.mean(x['attention_us'] for x in ctas),
            mean_attention_items_per_CTA=statistics.mean(x['attention_items'] for x in ctas),
            mean_lru_us_per_CTA=statistics.mean(x['lru_us'] for x in ctas),
            wait_fraction_of_attention=wait/attention,
            blocked_call_fraction=sum(x['blocked_calls'] for x in ctas)/max(1,sum(x['wait_calls'] for x in ctas)))
    # End timestamps identify the critical CTA per step; its item budget is useful but
    # does not isolate the cost of metadata loads or timers from compute.
    critical = [max(s['ctas'], key=lambda x: x['done_us']) for s in steps]
    summaries['critical_CTA'] = {key: statistics.mean(x[key] for x in critical)
        for key in ['wait_us','attention_us','attention_items','lru_us','before_items_us','last_attention_start_us','done_us']}
    records.append(dict(raw=str(f.relative_to(root)), raw_sha256=sha(f), setting=cfg['setting']['name'],
        diagnostic_event_mean_us=c['timings']['ours']['mean_us'],
        phase_means={key:statistics.mean(s[key] for s in steps) for key in steps[0] if key not in ['epoch','ctas']},
        summaries=summaries, steps=sorted(steps,key=lambda x:x['epoch']),
        max_abs_error=max(c['numerical_checks']['max_abs_error'].values())))
result = dict(goal_complete=False, candidate_adopted=False, classification='single-pass instrumented diagnostics only; no optimization or causal overhead claim',
    records=records, production_header_sha256=sha(root/'include/flashinfer/attention/mla_host_batch.cuh'),
    original_matrix_sha256=sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json'))
assert result['original_matrix_sha256'] == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
(p/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='steps'} for r in records],indent=2))
