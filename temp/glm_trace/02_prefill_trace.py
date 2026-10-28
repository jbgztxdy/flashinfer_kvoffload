# Purpose: real GLM-5.3 DSA indexer top-k traces on A800 (SGLang's DSA kernels need SM90): a
#          layer-by-layer BF16 prefill of the corpus documents, recording every indexer layer's
#          top-k for the positions decode would query (`--groups` windows of `--span` consecutive
#          positions over the second half of each document).
# Model: transformers glm_moe_dsa semantics, except the indexer RoPE, which follows
#        config.indexer_rope_interleave as SGLang does (transformers hard-codes non-interleaved).
#        FP8 128x128 block weights are dequantized to BF16 one layer at a time.
# Parallelism: torchrun, any number of ranks (1 GPU fits: one layer's BF16 weights at a time).
#        Every rank holds all hidden states; the indexer and the sparse attention split the rows
#        across ranks and all-gather; the 256 routed experts are split across ranks and their
#        outputs all-reduced. Attention runs in MLA's absorbed (latent, 512-wide) form.
# Output (rank 0): <out>/doc<i>.pt with, per indexer layer, group and position:
#        selections [indexer layers, groups, span, topk] int32 key positions within the document
#          (-1 padded), sorted by index score descending;
#        scores, same shape, fp32 index scores of those keys (-inf where padded), so scores[..., -1]
#          is the admission threshold when the row has >= topk keys;
#        row_stats [indexer layers, groups, span, 5] fp32 over all causal in-document keys:
#          n_valid, sum, sum of squares, max, top-k sum (mean/std/top-k share derive from these);
#        optional full_rows (--full-rows): full score row of each group's last position;
#        plus positions, group_starts, role (per document position, if the corpus has it), layer ids, source_layer (indexer layer used by each of the
#        78 layers), members (source sessions), doc index, loss and the run config.
# Usage: torchrun --nproc-per-node 8 02_prefill_trace.py --corpus corpus_64k.pt --out traces/
import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/WORK/PUBLIC/luyouy_work/rtx/models/zai-org--GLM-5.3")
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--docs", type=int, default=0, help="number of documents (0 = all)")
ap.add_argument("--tokens", type=int, default=0, help="truncate documents to this length (0 = full)")
ap.add_argument("--groups", type=int, default=16)
ap.add_argument("--span", type=int, default=1024, help="consecutive positions recorded per group")
ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (0 = all)")
ap.add_argument("--chunk", type=int, default=128, help="query rows per indexer chunk")
ap.add_argument("--attn-chunk", type=int, default=256, help="query rows per sparse-attention chunk")
ap.add_argument("--check", action="store_true", help="check chunked indexer/attention against dense references")
ap.add_argument("--indexer-interleave", type=int, choices=[0, 1], default=None,
                help="override config.indexer_rope_interleave (diagnostics)")
ap.add_argument("--dense", action="store_true", help="attention over all causal keys (diagnostics; short docs)")
ap.add_argument("--loss-all", action="store_true", help="save the next-token loss of every position to <out>/loss.pt")
ap.add_argument("--ref-layers", default="", help="comma-separated layers to compare against transformers' "
                "GlmMoeDsaDecoderLayer (1 rank, one doc, with --dense)")
ap.add_argument("--pos-offset", type=int, default=0, help="add to every RoPE position (diagnostics)")
ap.add_argument("--full-rows", action="store_true", help="also save the full score row of each group's last position")
args = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
cfg = json.load(open(f"{args.model}/config.json"))
H, D_NOPE, D_ROPE, D_V = cfg["num_attention_heads"], cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
KV_RANK, EPS = cfg["kv_lora_rank"], cfg["rms_norm_eps"]
IH, ID, TOPK = cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_topk"]
E, TOP_E, N_GROUP, TOPK_GROUP = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["n_group"], cfg["topk_group"]
SCALE_E, FIRST_DENSE = cfg["routed_scaling_factor"], cfg["first_k_dense_replace"]
THETA = cfg["rope_parameters"]["rope_theta"]
INDEXER_INTERLEAVE = cfg.get("indexer_rope_interleave", False) if args.indexer_interleave is None else bool(args.indexer_interleave)
TYPES = cfg["indexer_types"]
N_LAYERS = args.layers or cfg["num_hidden_layers"]
assert E % world == 0
E_LOCAL = E // world
SM_SCALE = (D_NOPE + D_ROPE) ** -0.5

index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
_handles = {}
REF_LAYERS = {int(v) for v in args.ref_layers.split(",") if v}


def ref_layer_check(L, h_in, h_out):
    """Run transformers' decoder layer L (eager, dense causal attention) on h_in with the same
    dequantized weights and report its relative error against this script's output h_out."""
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    assert world == 1 and args.dense
    conf = AutoConfig.from_pretrained(args.model)
    conf._attn_implementation, conf._experts_implementation = "eager", "eager"
    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(conf, L)
    layer.self_attn.indexer = None
    layer = layer.to_empty(device=dev).to(torch.bfloat16)
    p = f"model.layers.{L}."
    for name, t in layer.state_dict().items():
        if ".experts." in name:
            e = range(E)
            if name.endswith("gate_up_proj"):
                t.copy_(torch.stack([torch.cat([weight(f"{p}mlp.experts.{i}.gate_proj"), weight(f"{p}mlp.experts.{i}.up_proj")]) for i in e]))
            else:
                t.copy_(torch.stack([weight(f"{p}mlp.experts.{i}.down_proj") for i in e]))
        elif name.endswith(".weight"):
            t.copy_(weight(p + name[:-7]))
        else:
            t.copy_(raw(p + name).to(dev))
    if L >= FIRST_DENSE:  # the routing bias stays fp32, as in transformers
        layer.mlp.gate.e_score_correction_bias = vec(p + "mlp.gate.e_score_correction_bias").float()
    rot = M.GlmMoeDsaRotaryEmbedding(conf, device=dev)
    n = h_in.shape[0]
    pid = torch.arange(n, device=dev)[None]
    pe = rot(h_in[None], pid)
    mask = torch.full((n, n), torch.finfo(torch.bfloat16).min, device=dev).triu(1)[None, None].to(torch.bfloat16)
    allk = torch.arange(n, device=dev, dtype=torch.int32).expand(1, n, n)
    with torch.no_grad():
        out = layer(h_in[None], attention_mask=mask, position_ids=pid, position_embeddings=pe, prev_topk_indices=allk)[0][0]
    d, ref = (out.float() - h_out.float()), (out.float() - h_in.float())
    print(f"ref check layer {L}: |ours - transformers| / |layer delta| = {d.norm() / ref.norm():.3e}, "
          f"max abs {d.abs().max():.3e}, per-row worst rel {(d.norm(dim=-1) / ref.norm(dim=-1)).max():.3e}", flush=True)
    del layer
    torch.cuda.empty_cache()


def raw(name):
    f = index[name]
    if f not in _handles:
        _handles[f] = safe_open(f"{args.model}/{f}", "pt", device="cpu")
    return _handles[f].get_tensor(name)


def weight(name):
    """BF16 weight on the GPU, dequantizing FP8 with its 128x128 block scales."""
    w = raw(name + ".weight").to(dev)
    if w.dtype != torch.float8_e4m3fn:
        return w if w.dtype == torch.float32 else w.to(torch.bfloat16)
    s = raw(name + ".weight_scale_inv").to(dev)
    o, i = w.shape
    full = s.repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
    return (w.float() * full).to(torch.bfloat16)


def vec(name):
    return raw(name).to(dev)


def rms(x, w):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + EPS)).to(x.dtype) * w


def rope_tables(pos, dim):
    inv = 1.0 / (THETA ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float32) / dim))
    f = pos.float()[:, None] * inv[None]
    return f.cos(), f.sin()  # [T, dim/2]


def rope(x, cos, sin, interleave):
    """x [T, h, dim]; cos/sin [T, dim/2]. Interleaved pairs (x0,x1),(x2,x3).. or half-split."""
    c, s = cos[:, None], sin[:, None]
    if interleave:
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    else:
        x1, x2 = x[..., : x.shape[-1] // 2].float(), x[..., x.shape[-1] // 2 :].float()
    # Matches transformers' apply_rotary_pos_emb_interleave output layout (rotated halves concatenated).
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


def my_rows(T):
    """This rank's contiguous slice of rows, padded so every rank's slice has the same length."""
    per = -(-T // world)
    return per, torch.arange(rank * per, min(T, (rank + 1) * per), device=dev)


def gather_rows(local, per, T):
    """All-gather per-rank row slices (padded to `per`) into [T, ...]."""
    pad = torch.zeros((per,) + local.shape[1:], dtype=local.dtype, device=dev)
    pad[: len(local)] = local
    full = torch.empty((world * per,) + local.shape[1:], dtype=local.dtype, device=dev)
    dist.all_gather_into_tensor(full, pad)
    return full[:T]


def indexer_keys(L, x, cos, sin):
    p = f"model.layers.{L}.self_attn.indexer."
    k = F.layer_norm(x @ weight(p + "wk").T, (ID,), vec(p + "k_norm.weight").to(x.dtype),
                     vec(p + "k_norm.bias").to(x.dtype), 1e-6)[:, None]
    return torch.cat([rope(k[..., :D_ROPE], cos, sin, INDEXER_INTERLEAVE), k[..., D_ROPE:]], -1)[:, 0].float()


def indexer_scores(L, x, q_resid, k, rows, cos, sin, d0, d1):
    """Index scores [r, d1 - d0] of query rows against keys d0..d1 (before masking)."""
    p = f"model.layers.{L}.self_attn.indexer."
    q = (q_resid[rows] @ weight(p + "wq_b").T).view(len(rows), IH, ID)
    q = torch.cat([rope(q[..., :D_ROPE], cos[rows], sin[rows], INDEXER_INTERLEAVE), q[..., D_ROPE:]], -1).float()
    w = (x[rows].float() @ raw(p + "weights_proj.weight").to(dev).float().T) * IH ** -0.5
    sc = torch.einsum("rhd,td->rht", q, k[d0:d1]).mul_(ID ** -0.5).relu_()
    return torch.einsum("rh,rht->rt", w, sc)


def indexer_topk(L, x, q_resid, cos, sin, doc_bounds, T, full_rows=None):
    """Top-k key positions (within the row's document, causal) for every row, -1 padded, with
    their index scores (-inf padded) and per-row statistics; `full_rows` (global row ids) also
    returns those rows' whole score rows [len, T] (-inf outside the row's causal document)."""
    k = indexer_keys(L, x, cos, sin)
    per, mine = my_rows(T)
    out = torch.full((len(mine), TOPK), -1, dtype=torch.int32, device=dev)
    vals = torch.full((len(mine), TOPK), float("-inf"), dtype=torch.float32, device=dev)
    stats = torch.zeros((len(mine), 5), dtype=torch.float32, device=dev)
    full = torch.full((len(full_rows), T), float("-inf"), device=dev) if full_rows is not None else None
    for c0 in range(0, len(mine), args.chunk):
        rows = mine[c0 : c0 + args.chunk]
        lo = doc_bounds[torch.searchsorted(doc_bounds, rows, right=True) - 1]
        d0, d1 = int(lo.min()), int(rows.max()) + 1
        sc = indexer_scores(L, x, q_resid, k, rows, cos, sin, d0, d1)
        t = torch.arange(d0, d1, device=dev)
        ok = (t[None] <= rows[:, None]) & (t[None] >= lo[:, None])
        z = sc.where(ok, 0.0)
        stats[c0 : c0 + len(rows), :4] = torch.stack([ok.sum(-1).float(), z.sum(-1), z.pow(2).sum(-1),
                                                      sc.masked_fill(~ok, float("-inf")).amax(-1)], -1)
        sc.masked_fill_(~ok, float("-inf"))
        if full is not None:
            hit = (full_rows[:, None] == rows[None]).nonzero()
            full[hit[:, 0], d0:d1] = sc[hit[:, 1]]
        n = min(TOPK, sc.shape[1])
        val, top = sc.topk(n, -1)
        top = (top + d0 - lo[:, None]).int()
        top[val == float("-inf")] = -1
        out[c0 : c0 + len(rows), :n] = top
        vals[c0 : c0 + len(rows), :n] = val
        stats[c0 : c0 + len(rows), 4] = val.where(val > float("-inf"), 0.0).sum(-1)
    if full is not None:
        dist.all_reduce(full, dist.ReduceOp.MAX)  # each row lives on exactly one rank
    return gather_rows(out, per, T), gather_rows(vals, per, T), gather_rows(stats, per, T), full


def attention(L, x, cos, sin, doc_bounds, topk_all):
    """MLA (64 heads) with DSA sparse attention over each row's top-k keys, in the absorbed form:
    q_nope is mapped into the 512-wide latent through kv_b's key part, attention reads the
    normalized latent c and the shared rope key, and the latent output goes through kv_b's value
    part. Equivalent to the expanded form (transformers) up to rounding."""
    p = f"model.layers.{L}.self_attn."
    q_resid = rms(x @ weight(p + "q_a_proj").T, vec(p + "q_a_layernorm.weight"))
    ckv = x @ weight(p + "kv_a_proj_with_mqa").T
    c = rms(ckv[:, :KV_RANK], vec(p + "kv_a_layernorm.weight"))
    k_rope = rope(ckv[:, None, KV_RANK:], cos, sin, True)[:, 0]
    w_kv = weight(p + "kv_b_proj").view(H, D_NOPE + D_V, KV_RANK)
    w_uk, w_uv = w_kv[:, :D_NOPE], w_kv[:, D_NOPE:]
    w_qb = weight(p + "q_b_proj")
    T = x.shape[0]
    per, mine = my_rows(T)
    out = torch.empty((len(mine), H, D_V), dtype=x.dtype, device=dev)
    for c0 in range(0, len(mine), args.attn_chunk):
        rows = mine[c0 : c0 + args.attn_chunk]
        q = (q_resid[rows] @ w_qb.T).view(len(rows), H, D_NOPE + D_ROPE)
        q_lat = torch.einsum("rhd,hdc->rhc", q[..., :D_NOPE].float(), w_uk.float())
        q_rope = rope(q[..., D_NOPE:], cos[rows], sin[rows], True).float()
        lo = doc_bounds[torch.searchsorted(doc_bounds, rows, right=True) - 1]
        if args.dense:
            sel = torch.arange(int((rows - lo).max()) + 1, device=dev).expand(len(rows), -1)
            valid = sel <= (rows - lo)[:, None]
        else:
            sel = topk_all[rows].long()
            valid = sel >= 0
        g = lo[:, None] + sel.clamp(min=0)
        cg = c[g].float()  # [r, K, 512]
        s = torch.einsum("rhc,rkc->rhk", q_lat, cg) + torch.einsum("rhd,rkd->rhk", q_rope, k_rope[g].float())
        s = (s * SM_SCALE).masked_fill_(~valid[:, None], float("-inf")).softmax(-1)
        o_lat = torch.einsum("rhk,rkc->rhc", s, cg)
        out[c0 : c0 + len(rows)] = torch.einsum("rhc,hvc->rhv", o_lat, w_uv.float()).to(x.dtype)
    out = gather_rows(out.view(len(mine), -1), per, T)
    return out @ weight(p + "o_proj").T, q_resid


def dense_mlp(prefix, x):
    return (F.silu(x @ weight(prefix + "gate_proj").T) * (x @ weight(prefix + "up_proj").T)) @ weight(prefix + "down_proj").T


def moe(L, x):
    p = f"model.layers.{L}.mlp."
    scores = (x.float() @ raw(p + "gate.weight").to(dev).float().T).sigmoid()
    choice = scores + vec(p + "gate.e_score_correction_bias")
    if N_GROUP > 1:
        gs = choice.view(-1, N_GROUP, E // N_GROUP).topk(2, -1)[0].sum(-1)
        gm = torch.zeros_like(gs).scatter_(1, gs.topk(TOPK_GROUP, -1)[1], 1)
        choice = choice.masked_fill(~gm.repeat_interleave(E // N_GROUP, 1).bool(), float("-inf"))
    idx = choice.topk(TOP_E, -1, sorted=False)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE_E
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(rank * E_LOCAL, (rank + 1) * E_LOCAL):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        if len(tok):
            h = dense_mlp(f"{p}experts.{e}.", x[tok])
            out.index_add_(0, tok, h.float() * w[tok, slot, None])
    dist.all_reduce(out)
    return (out + dense_mlp(p + "shared_experts.", x).float()).to(x.dtype)


corpus = torch.load(args.corpus)
docs = corpus["tokens"][: args.docs or None]
if args.tokens:
    docs = [d[: args.tokens] for d in docs]
lens = [len(d) for d in docs]
doc_bounds = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=dev)
tokens = torch.cat(docs).to(dev).long()
pos = torch.cat([torch.arange(n) for n in lens]).to(dev) + args.pos_offset
T = len(tokens)
starts, want = [], []
for i, n in enumerate(lens):
    st = torch.linspace(n // 2, n - args.span, args.groups).long()
    starts.append(st)
    for s in st:
        want.append(int(doc_bounds[i]) + torch.arange(int(s), int(s) + args.span))
want = torch.cat(want).to(dev)
last = want.view(-1, args.span)[:, -1] if args.full_rows else None  # each group's last position
cos, sin = rope_tables(pos, D_ROPE)
hs = F.embedding(tokens, raw("model.embed_tokens.weight").to(dev))
records, layer_ids, source_layer, topk_all = [], [], [], None
t0 = time.time()
for L in range(N_LAYERS):
    h_in = hs.clone() if L in REF_LAYERS else None
    x = rms(hs, vec(f"model.layers.{L}.input_layernorm.weight"))
    if TYPES[L] == "full":
        q_resid = rms(x @ weight(f"model.layers.{L}.self_attn.q_a_proj").T,
                      vec(f"model.layers.{L}.self_attn.q_a_layernorm.weight"))
        topk_all, score_all, stats_all, full = indexer_topk(L, x, q_resid, cos, sin, doc_bounds, T, last)
        if rank == 0:
            records.append((topk_all[want].cpu(), score_all[want].cpu(), stats_all[want].cpu(),
                            full.cpu() if full is not None else None))
            layer_ids.append(L)
        if args.check and L == 0 and rank == 0:
            # Dense reference on the first document's last 256 rows: same scores without chunking.
            k = indexer_keys(L, x, cos, sin)
            rows = torch.arange(lens[0] - 256, lens[0], device=dev)
            ref = indexer_scores(L, x, q_resid, k, rows, cos, sin, 0, lens[0])
            ref.masked_fill_(torch.arange(lens[0], device=dev)[None] > rows[:, None], float("-inf"))
            ref_top = ref.topk(TOPK, -1).indices
            got = topk_all[rows].long()
            overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / TOPK for a, b in zip(ref_top, got)])
            print(f"check layer 0: chunked vs dense top-k overlap mean {overlap.mean():.4f} min {overlap.min():.4f}", flush=True)
            # Recorded scores are the dense scores at the recorded positions, sorted descending.
            ref_at = ref.gather(1, got.clamp(min=0)).masked_fill(got < 0, float("-inf"))
            fin = ref_at > float("-inf")
            serr = ((ref_at - score_all[rows]).abs() / ref.where(ref > float("-inf"), 0.0).abs().amax(-1, keepdim=True))[fin].max()
            desc = bool((score_all[rows][:, 1:] <= score_all[rows][:, :-1]).all())
            nv = (ref > float("-inf")).sum(-1).float()
            print(f"check layer 0: recorded vs dense scores max rel error {serr:.2e}, sorted {desc}, "
                  f"n_valid ok {bool((nv == stats_all[rows, 0]).all())}, -1 iff -inf "
                  f"{bool(((got < 0) == (score_all[rows] == float('-inf'))).all())}", flush=True)
    if args.check and L == 0:
        # Absorbed vs expanded sparse attention on a few rows of the first document.
        a_abs, _ = attention(L, x, cos, sin, doc_bounds, topk_all)
        if rank == 0:
            p = f"model.layers.{L}.self_attn."
            qr = rms(x @ weight(p + "q_a_proj").T, vec(p + "q_a_layernorm.weight"))
            q = (qr @ weight(p + "q_b_proj").T).view(-1, H, D_NOPE + D_ROPE)
            ckv = x @ weight(p + "kv_a_proj_with_mqa").T
            kv = (rms(ckv[:, :KV_RANK], vec(p + "kv_a_layernorm.weight")) @ weight(p + "kv_b_proj").T).view(-1, H, D_NOPE + D_V)
            kr = rope(ckv[:, None, KV_RANK:], cos, sin, True)[:, 0]
            rows = torch.arange(lens[0] - 64, lens[0], device=dev)
            g = topk_all[rows].long()
            sc = torch.einsum("rhd,rkhd->rhk", q[rows, :, :D_NOPE].float(), kv[g][..., :D_NOPE].float())
            sc += torch.einsum("rhd,rkd->rhk", rope(q[rows, :, D_NOPE:], cos[rows], sin[rows], True).float(), kr[g].float())
            o = torch.einsum("rhk,rkhd->rhd", (sc * SM_SCALE).softmax(-1), kv[g][..., D_NOPE:].float())
            ref = o.reshape(len(rows), -1).to(x.dtype) @ weight(p + "o_proj").T
            err = (ref.float() - a_abs[rows].float()).abs().max() / ref.float().abs().max()
            print(f"check layer 0: absorbed vs expanded attention, max rel error {err:.2e}", flush=True)
    source_layer.append(layer_ids[-1] if rank == 0 else -1)
    a, _ = attention(L, x, cos, sin, doc_bounds, topk_all)
    hs = hs + a
    x = rms(hs, vec(f"model.layers.{L}.post_attention_layernorm.weight"))
    hs = hs + (dense_mlp(f"model.layers.{L}.mlp.", x) if L < FIRST_DENSE else moe(L, x))
    if h_in is not None:
        ref_layer_check(L, h_in, hs)
    _handles.clear()
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"layer {L:2d} ({TYPES[L]}) at {time.time() - t0:6.0f} s, mean |h| {hs.float().norm(dim=-1).mean():.1f}, "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
if rank == 0:
    loss = float("nan")
    if N_LAYERS == cfg["num_hidden_layers"]:
        # Sanity: next-token loss over the recorded windows (a reading model gives a low loss).
        lm = raw("lm_head.weight").to(dev)
        nxt = want + 1
        ok = (nxt < T) & (torch.searchsorted(doc_bounds, nxt, right=True) == torch.searchsorted(doc_bounds, want, right=True))
        tok_loss = torch.full((len(want),), float("nan"), device=dev)
        idx = ok.nonzero()[:, 0]
        for c0 in range(0, len(idx), 1024):
            r = want[idx[c0 : c0 + 1024]]
            logits = (rms(hs[r], vec("model.norm.weight")) @ lm.T).float()
            tok_loss[idx[c0 : c0 + 1024]] = F.cross_entropy(logits, tokens[r + 1], reduction="none")
        loss = tok_loss[ok].mean().item()
        if args.loss_all:
            every = torch.full((T,), float("nan"), device=dev)
            pred = torch.full((T, 5), -1, dtype=torch.int32, device=dev)
            nxt_ok = torch.ones(T, dtype=torch.bool, device=dev)
            nxt_ok[doc_bounds[1:] - 1] = False
            r_all = nxt_ok.nonzero()[:, 0]
            for c0 in range(0, len(r_all), 1024):
                r = r_all[c0 : c0 + 1024]
                logits = (rms(hs[r], vec("model.norm.weight")) @ lm.T).float()
                every[r] = F.cross_entropy(logits, tokens[r + 1], reduction="none")
                pred[r] = logits.topk(5, -1).indices.int()
            Path(args.out).mkdir(parents=True, exist_ok=True)
            torch.save(dict(loss=every.cpu(), pred_top5=pred.cpu(), tokens=tokens.cpu(), doc_bounds=doc_bounds.cpu()),
                       Path(args.out) / "loss.pt")
            print("next-token loss per 256 positions:", " ".join(f"{v:.2f}" for v in every[: T // 256 * 256].view(-1, 256).nanmean(-1).tolist()),
                  flush=True)
        print("next-token loss per group:", " ".join(f"{v:.3f}" for v in tok_loss.view(-1, args.span).nanmean(-1).tolist()),
              flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel, scr, sts = (torch.stack([r[j] for r in records]) for j in range(3))
    per_doc = args.groups * args.span
    nl, G, S = len(layer_ids), args.groups, args.span
    for i in range(len(docs)):
        d = slice(i * per_doc, (i + 1) * per_doc)
        rec = dict(selections=sel[:, d].reshape(nl, G, S, TOPK).contiguous(),
                   scores=scr[:, d].reshape(nl, G, S, TOPK).contiguous(),
                   row_stats=sts[:, d].reshape(nl, G, S, 5).contiguous(),
                   row_stats_fields=["n_valid", "sum", "sumsq", "max", "topk_sum"],
                   positions=starts[i][:, None] + torch.arange(S), group_starts=starts[i],
                   layers=layer_ids, source_layer=source_layer, topk=TOPK, doc=i, tokens=lens[i],
                   members=corpus["members"][i], corpus=args.corpus, config=vars(args), model=args.model,
                   meta=corpus["meta"][i] if "meta" in corpus else None,
                   role=corpus["role"][i][: lens[i]] if "role" in corpus else None,
                   role_names=corpus.get("role_names"),
                   loss=loss, layers_run=N_LAYERS)
        if args.full_rows:
            b0, b1 = int(doc_bounds[i]), int(doc_bounds[i + 1])
            rec["full_rows"] = torch.stack([r[3][i * G : (i + 1) * G, b0:b1] for r in records]).contiguous()
        torch.save(rec, out / f"doc{i}.pt")
    print(f"saved {len(docs)} docs, indexer layers {layer_ids}, next-token loss {loss:.3f}, {time.time() - t0:.0f} s",
          flush=True)
dist.destroy_process_group()
