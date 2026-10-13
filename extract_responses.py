#!/usr/bin/env python3
"""extract_responses.py — dump bench JSONL to per-prompt text files and
extract exec_check Python snippets for objective verification."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(HERE, "results", "bench_minicpm5_20261012.jsonl")) if x.strip()]
spec = json.load(open(os.path.join(HERE, "prompts_abcd.json")))
exec_ids = {p["id"] for p in spec["prompts"] if p.get("exec_check")}

rdir = os.path.join(HERE, "results", "responses")
edir = os.path.join(HERE, "results", "exec")
os.makedirs(rdir, exist_ok=True)
os.makedirs(edir, exist_ok=True)

for r in rows:
    with open(os.path.join(rdir, r["id"] + ".txt"), "w") as f:
        f.write(f"id: {r['id']}  suite: {r['suite']}  cat: {r['category']}  ({r['label']})\n")
        f.write(f"completion_tokens: {r.get('completion_tokens')}  gen_tok_s: {r.get('gen_tok_s')}  "
                f"pp_tok_s: {r.get('pp_tok_s')}  total_s: {r.get('total_s')}  finish: {r.get('finish_reason')}\n")
        if r.get("expected_tool"):
            f.write(f"expected_tool: {r['expected_tool']}\n")
        if r.get("tool_calls"):
            f.write("tool_calls: " + json.dumps(r["tool_calls"]) + "\n")
        f.write("=" * 70 + "\n")
        f.write(r.get("text", ""))

for r in rows:
    if r["id"] not in exec_ids:
        continue
    text = r.get("text", "")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    if not blocks:
        blocks = re.findall(r"```\n(.*?)```", text, re.S)
    if not blocks:
        print(f"{r['id']}: NO CODE BLOCK FOUND")
        continue
    code = max(blocks, key=len)  # biggest block = the implementation
    with open(os.path.join(edir, r["id"] + ".py"), "w") as f:
        f.write(code)
    print(f"{r['id']}: extracted {len(code)} chars")

print(f"\n{len(rows)} responses -> {rdir}")