# Staged versus in-place reads on the GLM-5.3 trace

This compares the existing production modes without modifying the kernel or
replacing any cell in the original 5,304-case staged matrix.

Scope: MTP8, 16/64/128 requests, 4,096 LRU slots/request, full 1,024-position
windows, three cold-start passes, arithmetic means including the first step.
Links: 100 GB/s + 0/20 us, 50 GB/s + 20 us, physical NVLink. Two paths produce
24 formal route cases and two short correctness cases.

The benchmark copies the audited matrix benchmark, changing only the route
selector, output labels, shared benchmark lock and post-event miss diagnostic.
In-place remap entries retain positive far page ids and an owner bit, unlike
staged entries. The diagnostic masks the owner bit and counts remote references
outside the kernel events.

From the repository root, run run.py and then report.py in this directory.
Check existing process state first. The runner waits for foreign GPU0/1 users
and the observed profiler controller to finish. A first launch detected a foreign
process before CUDA allocation and failed without timing; it is retained under
failures/ and never enters the report.

report.py audits accepted means, pass lengths, cache-miss diagnostics, numerical
check records, queue segments, source snapshots and original trace bindings and
kernel source SHA. It writes routes.csv, summary.json and the report at
docs/mla_host_glm53_route_comparison.md. Raw samples and source snapshots remain.

Estimated bytes reflect logical unique-page or reference counts, not hardware
traffic counters; they exclude line/protocol effects. The all-local reference
comes from the original matched-shape normal-build complete-window replay.

The original raw staging_rows field describes the retained shared buffer plan.
For host_stage=False the wrapper sets internal staging capacity to zero; the
derived effective_kernel_staging_rows column explicitly records this distinction.
