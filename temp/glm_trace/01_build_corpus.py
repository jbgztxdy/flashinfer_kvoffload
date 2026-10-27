# Purpose: long agent-like documents for GLM-5.3 indexer top-k traces, from claudecode-trace.
#          Each session contributes its longest request (system + tools + messages, which already
#          contain every earlier turn); sessions are concatenated in a fixed shuffled order, with
#          their chat templates, into documents of up to --tokens tokens.
# Usage: python 01_build_corpus.py --tokens 65536 --docs 4 --out corpus.pt
import argparse, glob, json, os, random, sqlite3
import torch
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default="/WORK/PUBLIC/luyouy_work/rtx/yjw/data/claudecode-trace")
ap.add_argument("--model", default="/WORK/PUBLIC/luyouy_work/rtx/models/zai-org--GLM-5.3")
ap.add_argument("--tokens", type=int, default=65536)
ap.add_argument("--docs", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)

def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        t = part.get("type")
        if t == "text": out.append(part["text"])
        elif t == "tool_use": out.append("<tool_call>" + json.dumps(part.get("input"), ensure_ascii=False) + "</tool_call>")
        elif t == "tool_result": out.append("<tool_response>" + text_of(part.get("content")) + "</tool_response>")
    return "\n".join(out)

sessions = {}
for f in glob.glob(os.path.join(args.trace, "**/traces.sqlite"), recursive=True):
    con = sqlite3.connect(f)
    for sid, pt, body in con.execute("select session_id, prompt_token_count, request_body_json from requests"):
        key = (os.path.relpath(os.path.dirname(f), args.trace), sid)
        if key not in sessions or (pt or 0) > sessions[key][0]:
            sessions[key] = (pt or 0, body)
pieces = []
for key in sorted(sessions):
    b = json.loads(sessions[key][1])
    system = b.get("system")
    system = text_of(system) if not isinstance(system, str) else system
    msgs = [{"role": "system", "content": system + "\n# Tools\n" + json.dumps(b.get("tools", []), ensure_ascii=False)}] if system or b.get("tools") else []
    msgs += [{"role": m["role"], "content": text_of(m["content"])} for m in b.get("messages", [])]
    ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)["input_ids"])
    pieces.append((key, ids))
random.Random(args.seed).shuffle(pieces)
docs, cur, members = [], [], []
for key, ids in pieces * 4:  # reuse sessions in a new order if one pass is too short
    if len(docs) == args.docs: break
    take = ids[: args.tokens - len(cur)]
    cur += take; members.append("/".join(map(str, key)))
    if len(cur) >= args.tokens:
        docs.append((torch.tensor(cur, dtype=torch.int32), members)); cur, members = [], []
torch.save(dict(tokens=[d[0] for d in docs], members=[d[1] for d in docs], sessions=len(pieces),
                session_tokens=[len(p[1]) for p in pieces], tokens_per_doc=args.tokens), args.out)
print("sessions", len(pieces), "tokens per session p50", sorted(len(p[1]) for p in pieces)[len(pieces)//2],
      "total", sum(len(p[1]) for p in pieces), "docs", len(docs), "sessions per doc", [len(d[1]) for d in docs])
