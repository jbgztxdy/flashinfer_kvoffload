# Purpose: agent documents for GLM-5.3 DSA traces from nebius/SWE-rebench-openhands-trajectories
#          (OpenHands multi-turn SWE trajectories with OpenAI-style tool calls; CC-BY-4.0, from
#          hf-mirror). One trajectory per document, rendered with GLM-5.3's own chat template
#          (tools=, native <tool_call>/<tool_response> format), never concatenated. Keeps the
#          longest trajectories, truncated to --tokens.
# Output: tokens [docs] int32, role [docs] int8 per position (0 system, 1 user, 2 assistant,
#         3 tool, -1 template glue), members (trajectory ids), plus per-doc metadata.
# Usage: python 01b_build_corpus_swe.py --data rtx/data/swe-rebench-oh --docs 4 --tokens 65536 --out corpus_swe.pt
import argparse, json
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/WORK/PUBLIC/luyouy_work/rtx/data/swe-rebench-oh")
ap.add_argument("--model", default="/WORK/PUBLIC/luyouy_work/rtx/models/zai-org--GLM-5.3")
ap.add_argument("--tokens", type=int, default=65536)
ap.add_argument("--docs", type=int, default=4)
ap.add_argument("--shortlist", type=int, default=64, help="longest trajectories (by characters) to tokenize")
ap.add_argument("--resolved-only", type=int, default=1)
ap.add_argument("--show", action="store_true", help="print the rendered start of the first document")
ap.add_argument("--out", required=True)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
tools = json.load(open(f"{args.data}/tools.json"))
ROLES = {"system": 0, "user": 1, "assistant": 2, "tool": 3}


def messages(traj):
    out = []
    for m in traj:
        m = {k: v for k, v in m.items() if v is not None and k in ("role", "content", "tool_calls", "name", "tool_call_id")}
        for c in m.get("tool_calls") or []:
            if isinstance(c.get("function", {}).get("arguments"), str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
        m.setdefault("content", "")
        out.append(m)
    return out


def chars(traj):
    return sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"] or "") for c in m.get("tool_calls") or [])
               for m in traj)


# Shortlist by character count without materializing all rows.
f = pq.ParquetFile(f"{args.data}/trajectories.parquet")
cand = []
for b in f.iter_batches(batch_size=512, columns=["trajectory_id", "instance_id", "repo", "trajectory", "resolved"]):
    for r in b.to_pylist():
        if args.resolved_only and r["resolved"] != 1:
            continue
        cand.append((chars(r["trajectory"]), r))
    cand = sorted(cand, key=lambda x: -x[0])[: args.shortlist]

docs = []
for _, r in cand:
    msgs = messages(r["trajectory"])
    ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False)
    ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
    # Per-position role: render each prefix; tokens added by message i belong to its role.
    role = torch.full((len(ids),), -1, dtype=torch.int8)
    prev = 0
    for i in range(len(msgs)):
        p = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=True, add_generation_prompt=False)
        n = len(p["input_ids"] if hasattr(p, "keys") else p)
        role[prev:n] = ROLES[msgs[i]["role"]]
        prev = n
        if n >= args.tokens:
            break
    docs.append((len(ids), r, torch.tensor(ids, dtype=torch.int32), role))
    print(f"{r['trajectory_id']} {r['repo']} turns {len(msgs)} tokens {len(ids)}", flush=True)
docs.sort(key=lambda d: -d[0])
docs = docs[: args.docs]
assert all(d[0] >= args.tokens for d in docs), f"only {[d[0] for d in docs]} tokens; lower --tokens"
if args.show:
    print(tok.decode(docs[0][2][:3000]))
torch.save(dict(tokens=[d[2][: args.tokens] for d in docs], role=[d[3][: args.tokens] for d in docs],
                role_names=["system", "user", "assistant", "tool"],
                members=[[d[1]["trajectory_id"]] for d in docs],
                meta=[dict(instance_id=d[1]["instance_id"], repo=d[1]["repo"], full_tokens=d[0]) for d in docs],
                source="nebius/SWE-rebench-openhands-trajectories (hf-mirror), resolved only" if args.resolved_only else "",
                tokens_per_doc=args.tokens), args.out)
print("docs", [(d[1]["trajectory_id"], d[0]) for d in docs])
