"""GLM-5.3 kernel-only matrix: real sparse indices, actual cache payloads, arithmetic means."""
import argparse
import ctypes
import contextlib
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from resources_gpu3 import require_exclusive,snapshot

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
EXECUTED_BENCH_SOURCE=Path(__file__).read_bytes()
NUMA_POLICY=subprocess.check_output(['numactl','--show'],text=True)
assert next(line.split(':',1)[1].strip() for line in NUMA_POLICY.splitlines() if line.startswith('membind:'))=='1', NUMA_POLICY
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / "temp/h800"), str(ROOT / "temp"), str(ROOT / "temp/sim_link")]
import torch
import tiers
import emu
from flashinfer.jit import env as jit_env
VARIANT=os.environ.get("TUNE_VARIANT","base")
VARIANT_INCLUDE=(DEST/("include_"+VARIANT)).resolve()
assert VARIANT_INCLUDE.is_dir()
jit_env.FLASHINFER_INCLUDE_DIR=VARIANT_INCLUDE
from bench_glm52_host_batch import make_wrapper, reference
jit_env.FLASHINFER_INCLUDE_DIR=VARIANT_INCLUDE
if VARIANT in ("profile", "profile_final"):
    import inspect, textwrap
    import flashinfer.mla._core as mla_core
    cls=mla_core.BatchMLAPagedAttentionWrapper
    code=textwrap.dedent(inspect.getsource(cls._run_host_batch))
    code=code.replace("+ num_pools + rows,", "+ num_pools + rows + 2*128*128*16 + 2,")
    assert "2*128*128*16" in code
    namespace={}
    exec(code,mla_core.__dict__,namespace)
    cls._run_host_batch=namespace["_run_host_batch"]

META = json.loads((DEST / "experiment.json").read_text())
META["settings"].append(dict(name="native_hbm",tier="peer",emulated=False))
P = argparse.ArgumentParser()
P.add_argument("--mtp", type=int, required=True)
P.add_argument("--requests", type=int, required=True)
P.add_argument("--kind", choices=("floor", "snapshots", "nocache", "lru"), required=True)
P.add_argument("--settings", nargs="+", default=None)
P.add_argument("--smoke", action="store_true")
P.add_argument("--fetch-ctas",type=int,default=16)
P.add_argument("--host-item",type=int,default=64)
P.add_argument("--host-split",choices=("on","off"),default="on")
P.add_argument("--auto",action="store_true")
P.add_argument("--positions",type=int,default=1024)
P.add_argument("--route",choices=("staged","inplace"),default="staged")
P.add_argument("--fractions",nargs="+",type=float,default=None)
P.add_argument("--passes", type=int, default=META["passes"])
A = P.parse_args()
assert 0 < A.positions <= 1024 and A.positions % A.mtp == 0
assert A.passes > 0
if A.auto:
    assert os.environ.get("TUNE_VARIANT") in ("final", "base", "confirm_final", "confirm_base", "profile_final")
    if A.requests*A.mtp >= 432:
        A.fetch_ctas=24; A.host_split="off"
    else:
        A.fetch_ctas=16; A.host_split="on"
LOCK=(DEST.parent/"tune_general_20261031/gpu3_correctness.lock").open("w")
fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
# Fail before CUDA work if another process owns GPU3.
assert snapshot()['available'],snapshot()
RESOURCE_START=require_exclusive()
torch.set_num_threads(8)
torch.cuda.set_device(0)
torch.manual_seed(META["seed"])
DEV = torch.device("cuda", 0)
R, M, K, C, CAP = A.requests, A.mtp, META["topk"], 65536, META["capacity"]
ROWS, NNZ = R * M, R * M * K
STEPS = (16 if A.smoke else A.positions) // M
PASSES = A.passes
MODE = "ours"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    tmp.replace(path)


def hardware():
    fields="index,uuid,pci.bus_id,name,driver_version,memory.total,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    return subprocess.check_output(["nvidia-smi","--id=3","--query-gpu="+fields,
                                    "--format=csv,noheader"],text=True).strip().splitlines()


def sources():
    files={s: digest(ROOT / s) for s in (str(Path(__file__).relative_to(ROOT)),
            "temp/h800/tiers.py", "temp/sim_link/emu.py", "temp/sim_link/emu_gather.cu",
            "include/flashinfer/attention/mla_host_batch.cuh", "include/flashinfer/attention/mla.cuh",
            "include/flashinfer/attention/far_link_emu.cuh",
            str((DEST / "plans.cpp").relative_to(ROOT)),
            str((DEST / "experiment.json").relative_to(ROOT)),
            str((DEST / "resources_gpu3.py").relative_to(ROOT)),
            "temp/bench_glm52_host_batch.py",
            "flashinfer/mla/_core.py", "csrc/batch_mla_run.cu")}
    directory=DEST/"gpu3/source_snapshots";directory.mkdir(parents=True,exist_ok=True)
    files["include/flashinfer/attention/mla_host_batch.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla_host_batch.cuh")
    files["include/flashinfer/attention/mla.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla.cuh")
    files["include/flashinfer/attention/far_link_emu.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/far_link_emu.cuh")
    files[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest()
    for name,sha in files.items():
        target=directory/(sha+Path(name).suffix)
        if not target.exists():
            target.write_bytes(EXECUTED_BENCH_SOURCE if name==str(Path(__file__).relative_to(ROOT)) else ((VARIANT_INCLUDE/name.removeprefix("include/")) if name in ("include/flashinfer/attention/mla_host_batch.cuh","include/flashinfer/attention/mla.cuh","include/flashinfer/attention/far_link_emu.cuh") else (ROOT/name)).read_bytes())
    return files


caps = []
for path in sorted((ROOT / "temp/glm_trace/assistant64k").glob("doc*.pt")):
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d["replay_provenance"]["assistant_only"]
    for layer in range(len(d["layers"])):
        for group in range(len(d["group_starts"])):
            caps.append((d["selections"][layer, group], dict(file=str(path.relative_to(ROOT)),
                         layer=d["layers"][layer], window_start=int(d["group_starts"][group]),
                         source_trace_sha256=d["replay_provenance"]["source_trace_sha256"])))
random.Random(META["seed"]).shuffle(caps)
bindings = [caps[r % len(caps)][1] | {"request": r, "repeated_capture": r >= len(caps)} for r in range(R)]
# Identical prefix of requests at every batch size, and same request/layer/window throughout time.
positions = torch.stack([caps[r % len(caps)][0][:STEPS * M] for r in range(R)]).reshape(R, STEPS, M, K).permute(1,0,2,3).contiguous()
starts = torch.tensor([b["window_start"] for b in bindings]).view(1,R,1,1)
newest = starts + torch.arange(STEPS * M).view(STEPS,1,M,1)

SETTINGS = [s for s in META["settings"] if A.settings is None or s["name"] in A.settings]
if A.kind in ("floor","snapshots"):
    SETTINGS = [dict(name="floor", tier="peer", emulated=False)]
assert SETTINGS
assert all(s["name"] in ("floor","pcie","native_hbm") or s["name"].startswith(("g50_","g100_")) for s in SETTINGS), "GPU3 same-device far is not physical NVLink"
CACHED_PAGES = {}
CACHED_PLAN = None


def row_pages(near, fraction):
    source_positions=positions
    source_newest=newest
    if A.kind in ("nocache","snapshots"):
        selected_ticks=[x//M for x in META["no_lru_snapshots"] if x//M<STEPS] or [0]
        source_positions=positions[selected_ticks]
        source_newest=newest[selected_ticks]
    if fraction == 1:
        far = source_positions < source_newest
    elif fraction == 0:
        far = torch.zeros_like(source_positions, dtype=torch.bool)
    else:
        # Match the requested selected-reference fraction exactly, preserving the trace's
        # selected page identities and MTP overlap; mapping is per (request, position).
        far = torch.empty_like(source_positions, dtype=torch.bool)
        for r in range(R):
            gen = torch.Generator().manual_seed(META["seed"] * 7919 + r)
            flags = torch.rand(C, generator=gen) < fraction
            far[:,r] = flags[source_positions[:,r].long()] & (source_positions[:,r] < source_newest[:,r])
    owner = torch.arange(R).view(1,R,1,1)
    near_pages = (owner % min(R,128)) * C + source_positions
    pages = torch.where(far, near + owner * C + source_positions, near_pages).int().view(-1,ROWS,K)
    return pages, far.reshape(pages.shape[0],-1).sum(1).tolist()


def offline_lru(pages, near):
    libpath = DEST / "plans.so"
    if not libpath.exists() or libpath.stat().st_mtime < (DEST / "plans.cpp").stat().st_mtime:
        subprocess.run(["g++", "-fopenmp", "-O3", "-shared", "-fPIC", "-std=c++17", str(DEST / "plans.cpp"),
                        "-o", str(libpath)], check=True)
    library = ctypes.CDLL(str(libpath))
    fn = library.make_plan
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int64
    remap = torch.empty_like(pages)
    src, dst = torch.empty(pages.numel(), dtype=torch.int64), torch.empty(pages.numel(), dtype=torch.int64)
    offsets = torch.empty(STEPS+1, dtype=torch.int64)
    counts = torch.empty(STEPS,4,dtype=torch.int64)
    t = time.perf_counter()
    used = fn(*[ctypes.c_void_p(x.data_ptr()) for x in (pages,remap,src,dst,offsets,counts)],
              STEPS,R,M,K,C,near,CAP)
    return dict(remap=remap, src=src[:used].clone(), dst=dst[:used].clone(),
                offsets=offsets.tolist(), counts=counts.tolist(), cpu_plan_seconds=time.perf_counter()-t)


def chunk_metadata(pages, plan, first, last, near, floor=False):
    result = []
    for t in range(first,last):
        ids = pages[t].reshape(-1)
        if floor or plan is None:
            mask = ids >= near
            unique, inverse = torch.unique(ids[mask], return_inverse=True)
            mapped = ids.clone()
            mapped[mask] = (near + inverse).int()
            src, dst = (unique-near).long(), torch.arange(near,near+len(unique),dtype=torch.int64)
        else:
            a,b = plan["offsets"][t:t+2]
            src,dst = plan["src"][a:b],plan["dst"][a:b]
            mapped = plan["remap"][t].reshape(-1)
        result.append(dict(selected=ids.to(DEV), remap=mapped.to(DEV),
                           src=src.to(DEV), dst=dst.to(DEV), cpu_src=src))
    return result



def run_emu():
    assert os.environ.get('FLASHINFER_EXTRA_CUDAFLAGS')=='-DFLASHINFER_FAR_EMU'
    torch.manual_seed(META['seed']);near=R*C;slots=R*CAP;farstart=near+slots
    kv=tiers.TieredKV(farstart,R*C,R*CAP+NNZ,0,'peer',0);main=kv.main()
    for b in range(0,kv.near+kv.far,1<<20):main[b:b+(1<<20)].normal_()
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV);qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV);o=torch.empty_like(q)
    w=make_wrapper(ROWS,idx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    archive=DEST/'native_lru100_mapping.pt';record=torch.load(archive,weights_only=False);assert record['seed']==META['seed'] and record['farstart']==farstart
    meta=[{k:v.to(DEV) for k,v in d.items()} for d in record['meta']]
    packed=torch.empty((slots,1,576),dtype=torch.bfloat16,device=DEV)
    pages,_=row_pages(farstart,1.);selected=[x.reshape(-1).to(DEV) for x in pages]
    checkids=torch.tensor([0,512,1023],device=DEV)
    def prep(d):
        idx.copy_(d['indices']);n=d['src'].numel()
        if n:torch.index_select(main,0,d['src'],out=packed[:n]);main.index_copy_(0,d['dst'],packed[:n])
    def full():w.run(q,qr,main[...,:512],main[...,512:],out=o)
    result=dict(mapping_sha256=digest(archive),steps=record['summaries'],config=dict(gpu=3,requests=R,mtp=M,positions=A.positions,capacity=CAP,model='frozen local stall ps_per_byte=2000 empirical108CTA',passes=1),settings=[],resources_start=RESOURCE_START,complete=False,header_sha256=digest(VARIANT_INCLUDE/'flashinfer/attention/mla_host_batch.cuh'),far_sha256=digest(VARIANT_INCLUDE/'flashinfer/attention/far_link_emu.cuh'),harness_sha256=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest())
    target=DEST/'stall100_full_trace_gpu3.json';assert not target.exists()
    for label,bw,lat in [('disabled',None,0),('unlimited0',0,0),('g100_l0',100,0),('g100_l20000',100,20000)]:
        if bw is None:os.environ.pop('FLASHINFER_FAR_EMU',None)
        else:emu.kernel_env(bw,lat,main[farstart].data_ptr(),main[farstart].data_ptr()+kv.far*1152)
        prep(meta[0]);full();torch.cuda.synchronize()
        events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in meta]
        graph=torch.cuda.CUDAGraph(keep_graph=True);expected=[]
        with torch.cuda.graph(graph):
            for t,d in enumerate(meta):
                prep(d);events[t][0].record();full();events[t][1].record()
                if t in [0,16,127]:expected.append((t,o.index_select(0,checkids).float().clone()))
        graph.replay();torch.cuda.synchronize()
        for t,got in expected:torch.testing.assert_close(got,reference(main,q,qr,selected[t].view(ROWS,K),[0,512,1023]),atol=.01,rtol=.01)
        from nvlink_graph import inspect_graph
        launches=inspect_graph(graph);attention=[x for x in launches if 'BatchMLAPagedAttentionKernel' in x['kernel_name']];assert len(attention)==STEPS
        times=[a.elapsed_time(b)*1000 for a,b in events]
        result['settings'].append(dict(name=label,bandwidth_GBps=bw,latency_ns=lat,us=times,launches=attention[:1]));save(target,result);print('NATIVE EMU',label,sum(times)/STEPS,flush=True)
        del graph,events,expected
    result['libraries']=[dict(path=f,sha256=digest(f)) for f in {z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if '.so' in z and 'jit_' in z}];result['resources_end']=require_exclusive();result['sources']=sources();result['complete']=True;save(target,result)
run_emu()
