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
from resources import require_exclusive, snapshot

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
EXECUTED_BENCH_SOURCE=Path(__file__).read_bytes()
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / "temp/h800"), str(ROOT / "temp"), str(ROOT / "temp/sim_link")]
import torch
import tiers
import emu
from flashinfer.jit import env as jit_env
VARIANT=os.environ.get("TUNE_VARIANT","base")
VARIANT_INCLUDE=DEST.parent/"tune_qk_cursor_20261102/include_qk_cursor"
assert VARIANT_INCLUDE.is_dir()
jit_env.FLASHINFER_INCLUDE_DIR=VARIANT_INCLUDE
from bench_glm52_host_batch import make_wrapper, reference
jit_env.FLASHINFER_INCLUDE_DIR=VARIANT_INCLUDE
if VARIANT in ("profile_wait",):
    import inspect, textwrap
    import flashinfer.mla._core as mla_core
    cls=mla_core.BatchMLAPagedAttentionWrapper
    code=textwrap.dedent(inspect.getsource(cls._run_host_batch))
    code=code.replace("+ num_pools + rows,", "+ num_pools + rows + 2*128*128*32 + 2,")
    assert "2*128*128*32" in code
    namespace={}
    exec(code,mla_core.__dict__,namespace)
    cls._run_host_batch=namespace["_run_host_batch"]

META = json.loads((DEST / "experiment.json").read_text())
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
if VARIANT == "profile_wait":
    assert A.mtp == 8 and A.positions == 1024 and A.passes == 1 and A.kind == "lru"

assert 0 < A.positions <= 1024 and A.positions % A.mtp == 0
assert A.passes > 0
if A.auto:
    assert os.environ.get("TUNE_VARIANT") in ("final", "base", "confirm_final", "confirm_base", "profile_final")
    if A.requests*A.mtp >= 432:
        A.fetch_ctas=24; A.host_split="off"
    else:
        A.fetch_ctas=16; A.host_split="on"
LOCK=(ROOT/"temp/glm_trace/kernel_report_20261029/gpu01.lock").open("w")
fcntl.flock(LOCK,fcntl.LOCK_EX)
# The file lock can be released before the previous CUDA context is gone.
# Observe every process after acquiring it, before this process opens CUDA.
while not snapshot()['available']:
    print('WAITING_AFTER_LOCK', snapshot(), flush=True)
    time.sleep(2)
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
    return subprocess.check_output(["nvidia-smi","--id=0,1","--query-gpu="+fields,
                                    "--format=csv,noheader"],text=True).strip().splitlines()


def sources():
    files={s: digest(ROOT / s) for s in (str(Path(__file__).relative_to(ROOT)),
            "temp/h800/tiers.py", "temp/sim_link/emu.py", "temp/sim_link/emu_gather.cu",
            "include/flashinfer/attention/mla_host_batch.cuh", "include/flashinfer/attention/mla.cuh",
            "include/flashinfer/attention/far_link_emu.cuh",
            str((DEST / "plans.cpp").relative_to(ROOT)),
            str((DEST / "experiment.json").relative_to(ROOT)),
            str((DEST / "resources.py").relative_to(ROOT)),
            "temp/bench_glm52_host_batch.py",
            "flashinfer/mla/_core.py", "csrc/batch_mla_run.cu")}
    directory=DEST/"source_snapshots";directory.mkdir(exist_ok=True)
    files["include/flashinfer/attention/mla_host_batch.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla_host_batch.cuh")
    files["include/flashinfer/attention/mla.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla.cuh")
    files[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest()
    for name,sha in files.items():
        target=directory/(sha+Path(name).suffix)
        if not target.exists():
            target.write_bytes(EXECUTED_BENCH_SOURCE if name==str(Path(__file__).relative_to(ROOT)) else ((VARIANT_INCLUDE/name.removeprefix("include/")) if name in ("include/flashinfer/attention/mla_host_batch.cuh","include/flashinfer/attention/mla.cuh") else (ROOT/name)).read_bytes())
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


def run_setting(setting):
    global CACHED_PLAN
    require_exclusive()
    label = os.environ.get("TUNE_RUN_TAG","") + ("smoke_" if A.smoke else "") + VARIANT + f"_{A.route}_f{A.fetch_ctas}_i{A.host_item}_s{A.host_split}_{A.kind}_{setting['name']}_mtp{M}_b{R}"
    outpath = DEST / "raw" / (label + ".json")
    outpath.parent.mkdir(parents=True,exist_ok=True)
    previous=None
    if outpath.exists():
        existing=json.loads(outpath.read_text())
        if existing["complete"]:
            if A.kind!="nocache" or existing["configuration"].get("memcpy_path_version",1)>=2:
                print("skip complete",label,flush=True)
                return
            previous=existing
            print("repair only memcpy baseline",label,flush=True)
        failed=DEST/"failures";failed.mkdir(exist_ok=True)
        outpath.replace(failed/(outpath.stem+f"_{time.time_ns()}.json"))
    print("start",label,flush=True)
    torch.manual_seed(META["seed"])
    near = min(R,128)*C
    far = R*C
    staging = R*CAP+NNZ if A.kind=="lru" else NNZ
    kv=tiers.TieredKV(near,far,staging,0,setting["tier"],1)
    main,alias=kv.main(),kv.alias()
    for b in range(0,kv.near+kv.far,1<<20):
        main[b:b+(1<<20)].normal_()
    torch.cuda.synchronize()
    if setting.get("emulated"):
        assert "-DFLASHINFER_FAR_EMU" in os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS","")
        emu.kernel_env(setting["bandwidth_GBps"],setting["latency_ns"],main[kv.near].data_ptr(),
                       main[kv.near].data_ptr()+kv.far*1152)
    else:
        os.environ.pop("FLASHINFER_FAR_EMU",None)
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV)
    qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV)
    mi=torch.empty_like(idx)
    w=make_wrapper(ROWS,idx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    w.host_fetch_ctas=A.fetch_ctas;w.host_item_entries=A.host_item;w.host_stage=A.route=="staged"
    w.host_split=A.host_split=="on"
    # Test this batched staged kernel at every shape. The legacy one-row
    # implementation lacks complete virtual-link accounting.
    w._host_prefetch_shape=False
    # Avoid staging overflow: every unique selected miss must take the intended staged link path.
    w.host_staging_rows=max(64,NNZ)
    if A.kind=="lru":w.set_host_cache(torch.arange(R+1,dtype=torch.int32)*M,CAP)
    wm=make_wrapper(ROWS,mi,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    o=torch.empty_like(q);om=torch.empty_like(q)
    local=alias.view(-1,576)
    farview=main[kv.near:]
    packed=torch.empty((NNZ,1,576),dtype=torch.bfloat16,device=DEV)
    host=kv.far_cpu() if setting["tier"]=="host" else None
    hostpack=None
    fractions=[1.] if A.kind in ("lru","floor") else (A.fractions if A.fractions is not None else META["host_fractions"])
    result=dict(configuration=dict(kind=A.kind,setting=setting,requests=R,mtp=M,query_rows=ROWS,
                steps=STEPS,positions=STEPS*M,route=A.route,passes=PASSES,capacity=CAP if A.kind=="lru" else 0,topk=K,heads=64,
                aggregation=META["aggregation"],gpu_only=True,host_cpu_gather_timed=False,
                variant=VARIANT,ours_dispatch="batched_"+A.route+"_all_shapes",
                fetch_ctas=A.fetch_ctas,host_item_entries=A.host_item,host_split=A.host_split=="on",
                memcpy_path_version=2,
                host_pack_limit_GiB=int(os.environ.get("REPORT_HOST_PACK_LIMIT_GIB","8")),
                baseline_cpu_lru_planning_timed=False,staging_rows=NNZ),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),device=torch.cuda.get_device_name(0),
                hardware_start=hardware(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                resources_start=RESOURCE_START,
                source_sha256=sources(),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                bindings=bindings,residency=kv.residency,extra_cuda_flags=os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS",""),
                cases=[],complete=False)
    if previous:result["preserved_ours_source_sha256"]=previous["source_sha256"]
    check_rows=sorted({0,ROWS//2,ROWS-1})
    check_rows_gpu=torch.tensor(check_rows,dtype=torch.int64,device=DEV)
    for fraction in fractions:
        if fraction not in CACHED_PAGES:
            CACHED_PAGES[fraction]=row_pages(kv.near,fraction)
        pages,farrefs=CACHED_PAGES[fraction]
        if A.kind=="lru" and CACHED_PLAN is None:CACHED_PLAN=offline_lru(pages,kv.near)
        plan=CACHED_PLAN if A.kind=="lru" else None
        original_ticks=list(range(STEPS)) if A.kind in ("lru","floor") else [x//M for x in META["no_lru_snapshots"] if x//M<STEPS]
        ticks=list(range(pages.shape[0]))
        if not ticks:ticks=[0]
        modes=["local"] if A.kind in ("floor","snapshots") else ["ours","memcpy"]
        old_case=next((c for c in previous["cases"] if c["host_fraction_requested"]==fraction),None) if previous else None
        if previous:modes=["memcpy"]
        samples={mode:[[] for _ in ticks] for mode in modes}
        miss_samples=[[] for _ in ticks]
        ref_miss_samples=[[] for _ in ticks]
        guards=[];errors={mode:0. for mode in modes};cpu_gather=[]
        snapshot_meta=None
        if A.kind in ("nocache","snapshots"):
            snapshot_meta=chunk_metadata(pages,None,0,len(ticks),kv.near)
            if host is not None:
                for d in snapshot_meta:
                    n=d["cpu_src"].numel()
                    d["hostpack"]=torch.empty((n,1,576),dtype=torch.bfloat16,pin_memory=True)
                    begin=time.perf_counter_ns()
                    if n:torch.index_select(host,0,d["cpu_src"],out=d["hostpack"])
                    cpu_gather.append((time.perf_counter_ns()-begin)/1000)
        if A.kind=="lru":
            # Reuse immutable input/remap/copy metadata across all passes. This
            # removes repeated CPU planning/upload work, always outside events.
            estimate=pages.numel()*8+plan["src"].numel()*16
            free_bytes,_=torch.cuda.mem_get_info()
            kernel_reserve=(R*CAP+NNZ)*1152+far*12+NNZ*24+(1<<30)
            host_pack_bytes=plan["src"].numel()*1152 if host is not None else 0
            # Bound pinned memory even when the host has ample aggregate RAM:
            # NUMA-local allocation can exhaust one node with a full trace pack.
            host_pack_limit=int(os.environ.get("REPORT_HOST_PACK_LIMIT_GIB","8"))<<30
            if estimate < min(free_bytes*.65,max(0,free_bytes-kernel_reserve)):
                snapshot_meta=chunk_metadata(pages,plan,0,len(ticks),kv.near)
                print(label,"cached GPU metadata bytes",estimate,flush=True)
                if host is not None and host_pack_bytes < host_pack_limit:
                    for d in snapshot_meta:
                        n=d["cpu_src"].numel()
                        d["hostpack"]=torch.empty((n,1,576),dtype=torch.bfloat16,pin_memory=True)
                        begin=time.perf_counter_ns()
                        if n:torch.index_select(host,0,d["cpu_src"],out=d["hostpack"])
                        cpu_gather.append((time.perf_counter_ns()-begin)/1000)
        # Pass -1 checks payloads/output, is excluded from means. LRU state is reset each pass.
        check_steps=set(ticks if A.smoke else [ticks[i] for i in sorted({0,1 if len(ticks)>1 else 0,len(ticks)//2,len(ticks)-1})])
        graphs={}
        for repeat in range(-1,PASSES):
            order=modes if repeat%2==0 else list(reversed(modes))
            for mode in order:
                if A.kind=="lru" and mode=="ours":w.release_host_cache(torch.arange(R))
                for first in range(0,len(ticks),16):
                    graph_key=(mode,first)
                    if repeat>0 and graph_key in graphs:
                        cached=graphs[graph_key]
                        cached["graph"].replay();torch.cuda.synchronize()
                        for j,(start,end) in enumerate(cached["events"]):
                            samples[mode][first+j].append(start.elapsed_time(end)*1000)
                        if mode=="ours":
                            for j,v in enumerate(cached["diag"].cpu().tolist()):miss_samples[first+j].append(v)
                            for j,v in enumerate(cached["refs"].cpu().tolist()):ref_miss_samples[first+j].append(v)
                        for got,ref,t in cached["expected"]:
                            torch.testing.assert_close(got,ref,rtol=.01,atol=.01)
                        guards.append(dict(mode=mode,pass_index=repeat,first=first,last=first+len(cached["events"]),
                                           enqueue_ms=0.,guard_ms=0.,valid=True,timing_queue="cuda_graph"))
                        continue
                    chunk=ticks[first:first+16]
                    meta=[] if snapshot_meta is None else snapshot_meta[first:first+len(chunk)]
                    if snapshot_meta is None:
                        for t in chunk:
                            meta+=chunk_metadata(pages,plan,t,t+1,kv.near,floor=mode=="local")
                    if host is not None and mode!="ours":
                        for d in meta:
                            if "hostpack" in d:continue
                            n=d["cpu_src"].numel()
                            d["hostpack"]=torch.empty((n,1,576),dtype=torch.bfloat16,pin_memory=True)
                            begin=time.perf_counter_ns()
                            if n:torch.index_select(host,0,d["cpu_src"],out=d["hostpack"])
                            cpu_gather.append((time.perf_counter_ns()-begin)/1000)
                    graph_mode=A.kind=="lru" and repeat>=0 and os.environ.get("REPORT_GRAPH","1")=="1"
                    events=[(torch.cuda.Event(enable_timing=True,external=graph_mode),torch.cuda.Event(enable_timing=True,external=graph_mode)) for _ in chunk]
                    diag=torch.zeros(len(chunk),dtype=torch.int32,device=DEV)
                    diag_refs=torch.zeros(len(chunk),dtype=torch.int64,device=DEV)
                    # Warm the modules and initialize internal buffers without affecting a measured pass.
                    if first==0 and repeat==-1:
                        idx.copy_(meta[0]["selected"]);mi.copy_(meta[0]["remap"])
                        w.run(q,qr,main[...,:512],main[...,512:],out=o,host_page_start=kv.near)
                        wm.run(q,qr,alias[...,:512],alias[...,512:],out=om)
                        if A.kind=="lru" and mode=="ours":w.release_host_cache(torch.arange(R))
                    torch.cuda.synchronize()
                    # Guard guarantees all measured launches are queued before the GPU reaches them.
                    gs,ge=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    gs.record()
                    if repeat>=0 and not graph_mode:torch.cuda._sleep(int(os.environ.get("REPORT_GUARD_CYCLES","16000000")))
                    ge.record()
                    start_cpu=time.perf_counter()
                    expected=[]
                    graph=torch.cuda.CUDAGraph() if graph_mode else None
                    with torch.cuda.graph(graph) if graph_mode else contextlib.nullcontext():
                        for j,(t,d) in enumerate(zip(chunk,meta)):
                            ids=d["selected"] if mode=="ours" else d["remap"]
                            (idx if mode=="ours" else mi).copy_(ids)
                            n=d["src"].numel()
                            def copies():
                                if not n:return
                                direct=A.kind!="lru" or mode=="local"
                                target=alias[kv.near:kv.near+n] if direct else packed[:n]
                                if host is not None:target.copy_(d["hostpack"],non_blocking=True)
                                elif setting.get("emulated") and mode!="local":emu.gather(farview,d["src"],target)
                                else:torch.index_select(farview,0,d["src"],out=target)
                                if not direct:local.index_copy_(0,d["dst"],packed[:n,0])
                            if mode=="local":copies()
                            events[j][0].record()
                            if mode=="ours":
                                w.run(q,qr,main[...,:512],main[...,512:],out=o,host_page_start=kv.near)
                            else:
                                if mode=="memcpy":copies()
                                wm.run(q,qr,alias[...,:512],alias[...,512:],out=om)
                            events[j][1].record()
                            if mode=="ours":
                                if w._host_batch_buffers is not None:diag[j]=w._host_batch_buffers[5][1]
                                if A.kind=="lru":
                                    # Staged remap puts every cache hit below the cache_slots
                                    # boundary and every miss into the temporary staging tail.
                                    # Diagnostic reduction is after the measured end event.
                                    boundary=-(R*CAP+1)
                                    diag_refs[j]=((w._host_batch_buffers[3]<=boundary).sum() if A.route=="staged" else ((w._host_batch_buffers[3]>=0)&((w._host_batch_buffers[3]&0x3fffffff)>=kv.near)).sum())
                            if (repeat==-1 and t in check_steps) or (graph_mode and j==len(chunk)-1 and t in check_steps):
                                ref=reference(main,q,qr,d["selected"].view(ROWS,K),check_rows)
                                expected.append(((o if mode=="ours" else om).index_select(0,check_rows_gpu).float().clone(),ref,t))
                    if graph_mode:graph.replay()
                    enqueue_ms=(time.perf_counter()-start_cpu)*1000
                    torch.cuda.synchronize()
                    guard_ms=gs.elapsed_time(ge)
                    valid=graph_mode or guard_ms>=enqueue_ms+max(1.,enqueue_ms*.05)
                    guards.append(dict(mode=mode,pass_index=repeat,first=first,last=first+len(chunk),
                                       enqueue_ms=enqueue_ms,guard_ms=guard_ms,valid=valid,
                                       timing_queue="cuda_graph" if graph_mode else "eager_guard"))
                    if repeat>=0 and not valid:raise RuntimeError(f"CPU enqueue contaminated timing: {guards[-1]}")
                    for got,ref,t in expected:
                        torch.testing.assert_close(got,ref,rtol=.01,atol=.01,msg=lambda msg:f"{label} {mode} step {t}: {msg}")
                        errors[mode]=max(errors[mode],float((got-ref).abs().max()))
                    if repeat>=0:
                        diag_cpu=diag.cpu().tolist() if mode=="ours" else []
                        ref_cpu=diag_refs.cpu().tolist() if mode=="ours" and A.kind=="lru" else []
                        for j in range(len(chunk)):
                            samples[mode][first+j].append(events[j][0].elapsed_time(events[j][1])*1000)
                            if mode=="ours":miss_samples[first+j].append(diag_cpu[j])
                            if mode=="ours" and A.kind=="lru":ref_miss_samples[first+j].append(ref_cpu[j])
                    if graph_mode and snapshot_meta is not None and (host is None or host_pack_bytes < host_pack_limit):
                        graphs[graph_key]=dict(graph=graph,events=events,diag=diag,refs=diag_refs,expected=expected)
                    if snapshot_meta is not None and A.kind=="lru" and host is not None and host_pack_bytes >= host_pack_limit:
                        for d in meta:d.pop("hostpack",None)
                    del meta,events,diag,diag_refs,expected,graph
        unique=([plan["counts"][t][1] for t in ticks] if plan else
                [len(torch.unique(pages[t][pages[t]>=kv.near])) for t in ticks])
        if A.kind=="lru":oursmiss=[sum(x)/len(x) for x in miss_samples]
        else:oursmiss=unique
        cell=dict(host_fraction_requested=fraction if A.kind in ("nocache","snapshots") else None,
                  host_fraction_actual=sum(farrefs[t] for t in ticks)/(len(ticks)*NNZ),step_indices=original_ticks,
                  far_refs=[farrefs[t] for t in ticks],far_unique=unique,ours_unique_misses=oursmiss,
                  ours_unique_miss_samples=miss_samples if A.kind=="lru" else None,
                  ours_reference_miss_samples=ref_miss_samples if A.kind=="lru" else None,
                  ours_reference_cache_hit_rate=(1-sum(map(sum,ref_miss_samples))/PASSES/
                      max(1,sum(farrefs[t] for t in ticks))) if A.kind=="lru" else 0.,
                  ours_unique_cache_hit_rate=1-sum(oursmiss)/max(1,sum(unique)),
                  remote_traffic_savings_vs_raw_refs=1-sum(oursmiss)/max(1,sum(farrefs[t] for t in ticks)),
                  final_remote_unique_fraction=sum(oursmiss)/(len(ticks)*NNZ),
                  memcpy_plan_counts=plan["counts"] if plan else None,
                  cpu_plan_seconds=plan["cpu_plan_seconds"] if plan else 0.,
                  numerical_checks=dict(sampled_query_rows=check_rows,checked_steps=sorted(check_steps),max_abs_error=errors),
                  guards=guards,cpu_gather_us=cpu_gather,
                  timings={mode:dict(mean_us=sum(map(sum,v))/(len(v)*PASSES),per_step_samples_us=v,
                                    pass_mean_us=[sum(x[p] for x in v)/len(v) for p in range(PASSES)])
                           for mode,v in samples.items()})
        if old_case:
            cell["timings"]["ours"]=old_case["timings"]["ours"]
            cell["guards"]=[g for g in old_case["guards"] if g["mode"]=="ours"]+cell["guards"]
            cell["numerical_checks"]["max_abs_error"]["ours"]=old_case["numerical_checks"]["max_abs_error"]["ours"]
            cell["previous_memcpy_timings"]=old_case["timings"]["memcpy"]
        result["cases"].append(cell)
        print(label,"fraction",fraction,"mean_us",{m:round(v["mean_us"],3) for m,v in cell["timings"].items()},flush=True)
        save(outpath,result)
        del pages,plan,snapshot_meta,graphs
    if VARIANT in ("profile_wait",):
        state=w._host_batch_buffers[5]
        offset=state.numel()-2*128*128*32-2
        if (state.data_ptr()+offset*4)%8:offset+=1
        trace=state[offset:offset+2*128*128*32].view(torch.int64).cpu().tolist()
        result["wait_diagnostic_flat"]=trace
    result["hardware_end"]=hardware()
    result["resources_end"]=require_exclusive()
    result["complete"]=True;save(outpath,result)
    del main,alias,w,wm,q,qr,o,om,idx,mi,packed,local,farview,host,hostpack
    gc.collect();torch.cuda.synchronize();kv.close();torch.cuda.empty_cache()
    print("complete",label,flush=True)


for setting in SETTINGS:run_setting(setting)
