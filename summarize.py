#!/usr/bin/env python3
"""summarize_minicpm5.py — per-prompt metrics table + tool-call check."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(HERE, "results", "bench_minicpm5_20261012.jsonl")) if x.strip()]
spec = json.load(open(os.path.join(HERE, "prompts_abcd.json")))
keyed = {p["id"]: p for p in spec["prompts"]}

print(f"{'id':14} {'suite':5} {'finish':7} {'ct':>6} {'gen t/s':>8} {'ttft':>6}  {'tools':6} label")
for r in sorted(rows, key=lambda x: (x["suite"], x["id"])):
    tc = r.get("tool_calls") or []
    ttft = r.get("ttft_s") or 0
    print(f"{r['id']:14} {r['suite']:5} {r.get('finish_reason','?'):7} {r.get('completion_tokens',0):6} "
          f"{r.get('gen_tok_s',0) or 0:8} {ttft:6.2f}  {len(tc):<6} {r['label'][:40]}")

from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[r["suite"]].append(r)
print()
for s in "ABCD":
    if s not in by: continue
    rs = by[s]
    gens = [r["gen_tok_s"] for r in rs if r.get("gen_tok_s")]
    lens = [r.get("finish_reason") == "length" for r in rs]
    ct = [r.get("completion_tokens", 0) for r in rs]
    print(f"suite {s}: n={len(rs)} avg_gen={sum(gens)/len(gens):.2f} tok/s  "
          f"avg_ct={sum(ct)/len(ct):.0f}  capped={sum(lens)}/{len(rs)}")

print("\ntool-call detail (C-suite function calls):")
for r in rows:
    tc = r.get("tool_calls")
    exp = r.get("expected_tool")
    if tc is not None and len(tc) or exp:
        status = "OK" if (tc and exp and tc[0].get("name") == exp) else ("NONE" if not tc else "MISMATCH")
        print(f"  {r['id']:12} expected={exp or '-':12} got={[t.get('name') for t in (tc or [])]} [{status}]")
        if tc:
            args = json.dumps(tc[0].get("arguments"))[:110]
            print(f"               args: {args}")