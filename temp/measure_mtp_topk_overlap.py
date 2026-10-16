# Purpose: Measure how much the top-k KV sets of consecutive MTP query tokens overlap, using real
#          attention of DeepSeek-V2-Lite (MLA) as a proxy for a DSA indexer.
# Usage: python temp/measure_mtp_topk_overlap.py --device 3 --context 65536 \
#          --docs temp/mtp_overlap/docs/*.txt --output temp/mtp_overlap/overlap_64k.json
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

MODEL = "/mnt/data/mhy/model/deepseekv2-16b"


class Capture:
    """Score selected query rows against all earlier keys and keep each row's top-k positions."""

    def __init__(self, rows, topk, chunk):
        self.rows = rows
        self.topk = topk
        self.chunk = chunk
        self.selections = {}

    def __call__(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        output = sdpa_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        layer = module.layer_idx
        length = key.shape[2]
        positions = torch.arange(length, device=key.device)
        picked = []
        for begin in range(0, len(self.rows), self.chunk):
            rows = self.rows[begin : begin + self.chunk]
            weights = torch.zeros((len(rows), length), device=key.device)
            # One head at a time in FP32 keeps the long-context score buffer small.
            for head in range(query.shape[1]):
                scores = (query[0, head, rows].float() @ key[0, head].float().T) * scaling
                scores.masked_fill_(positions[None, :] > rows[:, None], float("-inf"))
                weights += scores.softmax(-1)
            # Mean attention probability across heads: one shared selection per token, like a DSA indexer.
            picked.append(weights.topk(self.topk, dim=-1).indices.int().cpu())
            del scores, weights
        self.selections[layer] = torch.cat(picked)
        return output


def streamed_forward(model, ids, device, mlp_chunk):
    """Run one decoder layer at a time on the GPU; the full model does not fit beside other jobs."""
    inner = model.model
    hidden = inner.embed_tokens(ids.cpu()).to(device)
    positions = torch.arange(ids.shape[1], device=device)[None]
    rotary = inner.rotary_emb.to(device)
    embeddings = rotary(hidden, position_ids=positions)
    for layer in inner.layers:
        layer.to(device)
        residual = hidden
        attended, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(hidden),
            attention_mask=None,
            position_embeddings=embeddings,
        )
        hidden = residual + attended
        del attended, residual
        for begin in range(0, hidden.shape[1], mlp_chunk):
            part = hidden[:, begin : begin + mlp_chunk]
            part += layer.mlp(layer.post_attention_layernorm(part))
        layer.to("cpu")
    return inner.norm.to(device)(hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--docs", nargs="+", type=Path, required=True)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--mtp", type=int, default=8)
    parser.add_argument("--groups", type=int, default=32, help="MTP groups per document")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--mlp-chunk", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda", args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    for layer in model.model.layers:
        for parameter in layer.parameters():
            parameter.data = parameter.data.pin_memory()
    length = args.context
    # Groups of MTP consecutive positions, evenly spread over the second half of the context.
    starts = torch.linspace(length // 2, length - args.mtp, args.groups).long()
    rows = (starts[:, None] + torch.arange(args.mtp)[None, :]).reshape(-1).to(device)
    capture = Capture(rows, args.topk, args.chunk)
    AttentionInterface.register("topk_capture", capture)
    model.set_attn_implementation("topk_capture")
    result = dict(
        configuration=vars(args)
        | {"docs": [str(d) for d in args.docs], "output": str(args.output), "model": MODEL},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        selection="per token: top-k of head-averaged softmax attention over keys at positions <= token",
        group_starts=starts.tolist(),
        documents=[],
    )
    for path in args.docs:
        ids = tokenizer(path.read_text(errors="ignore"), return_tensors="pt").input_ids
        if ids.shape[1] < length:
            raise ValueError(f"{path} has only {ids.shape[1]} tokens")
        ids = ids[:, :length].to(device)
        torch.cuda.reset_peak_memory_stats(device)
        capture.selections = {}
        begin = time.time()
        with torch.no_grad():
            hidden = streamed_forward(model, ids, device, args.mlp_chunk)
            # Next-token loss on the tail confirms the model still reads this context coherently.
            logits = model.lm_head(hidden[0, -1025:-1].cpu()).float()
            tail_loss = torch.nn.functional.cross_entropy(logits, ids[0, -1024:].cpu()).item()
            del hidden, logits
        torch.cuda.synchronize(device)
        selections = torch.stack([capture.selections[i] for i in sorted(capture.selections)])
        selections = selections.view(len(capture.selections), args.groups, args.mtp, args.topk)
        save = args.output.with_name(args.output.stem + f"_{path.stem}.pt")
        torch.save(dict(selections=selections, group_starts=starts), save)
        result["documents"].append(
            dict(
                name=path.stem,
                tokens=length,
                seconds=time.time() - begin,
                tail_loss=tail_loss,
                selections=str(save),
                peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            )
        )
        print(f"{path.stem}: loss={tail_loss:.3f} {time.time() - begin:.1f}s peak={result['documents'][-1]['peak_memory_gib']:.1f}GiB", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
