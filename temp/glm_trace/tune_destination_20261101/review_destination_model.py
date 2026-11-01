"""CPU structural oracle for destination remapping/publication, independent of attention math."""
import hashlib
import itertools
import json
from pathlib import Path
import random

p = Path(__file__).resolve().parent
rng = random.Random(20261101)
cases = []
for unique, cache_slots, admitted, references in itertools.product(
        [0, 1, 31, 64, 65, 257], [1, 8, 512], [0, 1, 7, 64], [1, 8]):
    count = min(unique, cache_slots, admitted)
    victims = rng.sample(range(cache_slots), count)
    victim_of = victims + [-1] * (unique - count)
    pages = [10000 + i for i in range(unique)]
    slot_page = [i for i in range(cache_slots)]
    resident = {i: i for i in range(cache_slots)}
    # Protected hits must never belong to the supplied evictable victim set.
    hits = [s for s in range(cache_slots) if s not in victims][:3]
    for slot in victims:
        resident.pop(slot_page[slot])
        slot_page[slot] = -1
    remap = [-(cache_slots + i) - 1 for _ in range(references) for i in range(unique)]
    for j, entry in enumerate(remap):
        i = -entry - 1 - cache_slots
        if victim_of[i] >= 0:
            remap[j] = -victim_of[i] - 1
    staging_ready, memory = set(), {}
    order = list(range(unique))
    rng.shuffle(order)
    for i in order:
        destination = victim_of[i] if victim_of[i] >= 0 else cache_slots + i
        if victim_of[i] >= 0:
            assert slot_page[destination] < 0
        # Represent the 72 distinct uint4 payload vectors; no ready publication
        # is permitted before the entire row is stored.
        payload = tuple((pages[i], vector) for vector in range(72))
        memory[destination] = payload
        if victim_of[i] >= 0:
            resident[pages[i]] = destination
            slot_page[destination] = pages[i]
        else:
            staging_ready.add(i)
    for j, entry in enumerate(remap):
        i = j % unique
        row = -entry - 1
        assert memory[row] == tuple((pages[i], vector) for vector in range(72))
        if row < cache_slots:
            assert slot_page[row] == pages[i] and resident[pages[i]] == row
        else:
            assert i in staging_ready
    assert all(slot_page[s] == s and resident[s] == s for s in hits)
    assert len(memory) == unique
    cases.append(dict(unique=unique, cache_slots=cache_slots, admitted=count, references=references))
record = dict(complete=True, cpu_only=True, cases=cases, case_count=len(cases),
              candidate_sha256=hashlib.sha256((p / 'final_destination.cuh').read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='32-bit remap and supplied unique victim map, all72vectors, protected hits, random publication order.',
              limitation='Victim map is supplied. Does not validate GPU ordering, LRU selection, attention math, bandwidth or performance.')
(p / 'destination_model_review.json').write_text(json.dumps(record, indent=2) + '\n')
print('DESTINATION_MODEL_OK', len(cases))
