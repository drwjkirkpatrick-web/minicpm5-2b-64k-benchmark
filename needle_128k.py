#!/usr/bin/env python3
"""needle_128k.py — deep-context needle retrieval at ~121K depth against the
LIVE maxed server (131072 ctx, DSpark, q8_0 KV). Needle at position ~0, filler
after, question last -> recall must span the full ~121K window.

Greedy (temp 0) per the needle-test rule. Filler reps computed from /tokenize
so the actual depth comes from MiniCPM's own tokenizer.
"""
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "needle_121k.json")
BASE = "http://127.0.0.1:8090"

NEEDLE = "The secret access word for the vault is PINECONE-42."
FILLER = ("The quarterly logistics review covers inventory turnover, "
          "carrier contracts, warehouse throughput and demand forecasts. ")
QUESTION = "\n\nWhat is the secret access word for the vault? Answer with just the word."


def tokenize(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                  json.dumps({"content": text}).encode(),
                                  {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


def main():
    target_prompt_tokens = 121500  # leaves ~9.6K headroom at 131072 ctx

    # measure filler packing on MiniCPM's tokenizer
    probe = FILLER * 100
    tok100 = tokenize(probe)
    per_line = tok100 / 100.0
    print(f"filler: {per_line:.2f} tokens/line (probe: {tok100} tok / 100 lines)")

    overhead = tokenize(NEEDLE + "\n\n" + QUESTION) + 4  # template wrapper margin
    reps = int((target_prompt_tokens - overhead) / per_line)
    est = overhead + reps * per_line
    print(f"reps={reps}  estimated prompt ~{est:.0f} tokens")

    prompt = NEEDLE + "\n\n" + FILLER * reps + QUESTION
    body = {
        "model": "minicpm5-2b-131k",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})

    print("sending needle prompt (long prompt processing, be patient)...", flush=True)
    t_send = time.time()
    ttft = None; ct = 0; pt = 0; parts = []; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            ch = j.get("choices")
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t_send
                    print(f"first token after {ttft:.1f}s (pp + ttft)", flush=True)
                parts.append(ch[0]["delta"]["content"])
            if j.get("usage"):
                usage = j["usage"]
    total = time.time() - t_send
    answer = "".join(parts).strip()
    pt = usage["prompt_tokens"]; ct = usage["completion_tokens"]
    pp_rate = pt / ttft if ttft else None
    ok = "PINECONE-42" in answer.upper()

    rec = {
        "id": "needle_121k", "model": "MiniCPM5-2B Q4_K_M",
        "config": "ctx 131072 native, KV q8_0, DSpark draft-dspark, greedy temp 0",
        "needle": "PINECONE-42", "grade": "PASS" if ok else "FAIL",
        "depth_prompt_tokens": pt, "answer": answer[:200],
        "pp_tok_s_at_depth": round(pp_rate, 1) if pp_rate else None,
        "ttft_s": round(ttft, 2) if ttft else None,
        "gen_tokens": ct, "total_s": round(total, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()