#!/usr/bin/env python3
"""maxtoken_probe.py — maxed long-form generation: 32768-token headroom,
sustained speed + repetition-loop watch. Detached runner."""
import json, time, urllib.request

body = {"model": "minicpm5-2b-131k",
        "messages": [{"role": "user", "content": "Write a complete, detailed long-form essay on the history, engineering and cultural significance of lighthouses. Aim for at least 3000 words with section headings."}],
        "max_tokens": 32768, "temperature": 0.3, "top_p": 0.95, "top_k": 40,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time()
parts = []
ct = 0
usage = None
finish = None
with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        j = json.loads(d)
        ch = j.get("choices")
        if ch and (ch[0].get("delta") or {}).get("content"):
            parts.append(ch[0]["delta"]["content"])
        if j.get("usage"):
            usage = j["usage"]
            finish = ch[0]["finish_reason"] if ch else None

text = "".join(parts)
total = time.time() - t0
ct = usage["completion_tokens"] if usage else 0
rec = {"prompt": "lighthouse long essay", "max_tokens_set": 32768,
       "completion_tokens": ct, "finish_reason": finish,
       "total_s": round(total, 1), "gen_tok_s": round(ct / total, 2) if total else None,
       "chars": len(text)}
open("results/maxtoken_probe.txt", "w").write(text)
json.dump(rec, open("results/maxtoken_probe.json", "w"), indent=1)
print(json.dumps(rec, indent=1), flush=True)