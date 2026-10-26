# Purpose: find bgload settings that leave a target far-read bandwidth for GPU0 reading GPU1 over
#          NVLink, and check that the background traffic does not slow all-HBM attention.
#          For each setting: SM random gather of 1152-byte rows (the kernel's access pattern) and
#          contiguous DMA from GPU1, plus the native MLA kernel with every row in GPU0 HBM.
# Usage: python temp/sim_link/calib.py --targets 100 50 [--ctas 32] [--out temp/sim_link/calib.json]
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch  # noqa: E402
import tiers  # noqa: E402
from bench_glm52_host_batch import make_wrapper  # noqa: E402

BG = str(Path(__file__).with_name("bgload"))


class Background:
    """bgload running for the duration of a `with` block (none if ctas == 0)."""

    def __init__(self, ctas, sleep_ns, threads=256):
        self.args = (ctas, threads, sleep_ns)

    def __enter__(self):
        self.proc = None
        if self.args[0]:
            self.proc = subprocess.Popen([BG, *map(str, self.args), "0"], stdout=subprocess.PIPE, text=True)
            time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            out, _ = self.proc.communicate(timeout=30)
            self.rate = float(out.split(":")[1].split("GB/s")[0])
        else:
            self.rate = 0.0


def timed(fn, n=7):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / 1000)
    return sorted(ts)[n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", type=float, default=[100, 50])
    ap.add_argument("--ctas", type=int, default=32)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--out", default="temp/sim_link/calib.json")
    args = ap.parse_args()
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    rows = 1 << 20  # 1.2 GB of far rows read per probe
    kv = tiers.TieredKV(rows, 2 * rows, rows, 0, "peer", 1)
    main_kv, alias = kv.main(), kv.alias()
    for b in range(0, kv.near + kv.far, 2**20):
        main_kv[b:b + 2**20].normal_()
    ids = torch.randperm(kv.far, device=dev)[:rows] + kv.near
    dst = alias[kv.near:kv.near + rows]

    def gather():
        torch.index_select(main_kv, 0, ids, out=dst)

    def dma():
        dst.copy_(main_kv[kv.near:kv.near + rows])

    # All-HBM attention: 128 requests x MTP8, every selected row near.
    q_rows, topk = 1024, 2048
    idx = torch.randint(0, kv.near, (q_rows * topk,), dtype=torch.int32, device=dev)
    w = make_wrapper(q_rows, idx, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
    q = torch.randn(q_rows, 64, 512, dtype=torch.bfloat16, device=dev)
    qp = torch.randn(q_rows, 64, 64, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(q)

    def attn():
        w.run(q, qp, main_kv[..., :512], main_kv[..., 512:], out=out)

    def measure(ctas, sleep_ns):
        with Background(ctas, sleep_ns) as bg:
            g = rows * tiers.ROW_BYTES / timed(gather) / 1e9
            d = rows * tiers.ROW_BYTES / timed(dma) / 1e9
            a = timed(attn) * 1e6
        return dict(ctas=ctas, sleep_ns=sleep_ns, background_GBps=bg.rate, gather_GBps=g, dma_GBps=d, attn_hbm_us=a)

    base = measure(0, 0)
    print("no background:", base, flush=True)
    results = dict(baseline=base, targets={})
    for target in args.targets:
        lo, hi = 0, 200000  # sleep_ns: larger sleep = less background = more residual bandwidth
        best = None
        for _ in range(14):
            mid = (lo + hi) // 2
            m = measure(args.ctas, mid)
            print(f"target {target}: sleep {mid} ns -> gather {m['gather_GBps']:.1f} GB/s (bg {m['background_GBps']:.1f})", flush=True)
            if best is None or abs(m["gather_GBps"] - target) < abs(best["gather_GBps"] - target):
                best = m
            if abs(m["gather_GBps"] - target) <= args.tolerance * target:
                break
            if m["gather_GBps"] > target:
                hi = mid
            else:
                lo = mid
        # Repeat the chosen setting to report its spread.
        reps = [measure(best["ctas"], best["sleep_ns"]) for _ in range(3)]
        best["repeat_gather_GBps"] = [r["gather_GBps"] for r in reps]
        best["repeat_attn_hbm_us"] = [r["attn_hbm_us"] for r in reps]
        best["attn_slowdown"] = max(best["repeat_attn_hbm_us"]) / base["attn_hbm_us"]
        best["ok"] = (all(abs(g - target) <= args.tolerance * target for g in best["repeat_gather_GBps"])
                      and best["attn_slowdown"] < 1.02)
        results["targets"][str(target)] = best
        print(f"target {target}: {best}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
