#!/usr/bin/env python3
"""
bench_qwen35.py — A/B/C/D prompt suites against the live Qwen3.5-2B llama-server
(127.0.0.1:8090, 64K q8_0 KV, MTP, enable_thinking=false server-side).

Prompts: prompts_abcd.json (34 prompts, verbatim from prior suites).
Sampling: temp 0.3, top_p 0.95, top_k 40, min_p 0.0 — matches the original
llama-cli runs of suites A/B/C (temp 0.3 + llama.cpp defaults).

Measures per prompt (streaming, include_usage):
  ttft_s          time to first content/tool token (≈ prompt-processing time)
  pp_tok_s        prompt_tokens / ttft
  gen_tok_s       completion_tokens / (total - ttft)   [pure generation rate]
  eff_tok_s       completion_tokens / total             [incl. prompt processing]
Plus finish_reason, full text, assembled tool_calls, usage.

Incremental JSONL (resume-safe: rerun skips ids already recorded).

Usage:
  python3 bench_qwen35.py --suite a            # one suite (a|b|c|d)
  python3 bench_qwen35.py --ids a_code,d_fib  # specific prompts
  python3 bench_qwen35.py --smoke              # single smoke-test prompt
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(HERE, "prompts_abcd.json")
RESULTS_FILE = os.path.join(HERE, "results", "bench_minicpm5_20261012.jsonl")
META_FILE = os.path.join(HERE, "results", "meta_minicpm5_20261012.json")
BASE = "http://127.0.0.1:8090"

# Tool schemas for suite C function-call prompts (mirror the expected tools).
TOOL_SCHEMAS = {
    "web_search": {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query"}},
                       "required": ["query"]}}},
    "terminal": {"type": "function", "function": {
        "name": "terminal",
        "description": "Execute a shell command on the user's system",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                       "required": ["command"]}}},
    "write_file": {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file on disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"},
                                      "content": {"type": "string", "description": "File content"}},
                       "required": ["path", "content"]}}},
    "read_file": {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path to read"}},
                       "required": ["path"]}}},
}


def server_snapshot():
    snap = {}
    try:
        with urllib.request.urlopen(BASE + "/props", timeout=5) as r:
            props = json.load(r)
        snap["model_path"] = props.get("model_path")
        snap["n_ctx"] = props.get("n_ctx")
    except Exception as e:
        snap["props_error"] = str(e)
    r = subprocess.run(["pgrep", "-f", "llama-server.*--alias qwen3.5-2b"],
                       capture_output=True, text=True)
    pids = r.stdout.split()
    if pids:
        pid = pids[0]
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        snap["rss_gb"] = round(int(line.split()[1]) / 1048576, 2)
            snap["pid"] = pid
            snap["oom_score_adj"] = open(f"/proc/{pid}/oom_score_adj").read().strip()
        except OSError:
            pass
    snap["free_m"] = subprocess.run(["sh", "-c", "free -m | awk 'NR==2{print $7}'"],
                                     capture_output=True, text=True).stdout.strip()
    snap["gdm"] = subprocess.run(["systemctl", "is-active", "gdm"],
                                 capture_output=True, text=True).stdout.strip()
    snap["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return snap


def stream_chat(prompt, tools=None, max_tokens=1024, timeout=900):
    body = {
        "model": "qwen3.5-2b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3, "top_p": 0.95, "top_k": 40,
        "min_p": 0.0, "repeat_penalty": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text_parts = []
    tool_calls = {}
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    text_parts.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.time() - t0
    gen_time = (total - ttft) if ttft else total
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    return {
        "text": "".join(text_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "finish_reason": finish,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 2),
        "pp_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
        "gen_tok_s": round(ct / gen_time, 2) if (ct and gen_time > 0) else None,
        "eff_tok_s": round(ct / total, 2) if (ct and total > 0) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["a", "b", "c", "d"],
                    help="run all prompts of one suite")
    ap.add_argument("--ids", help="comma-separated prompt ids to run")
    ap.add_argument("--smoke", action="store_true",
                    help="run a single smoke-test prompt (d_r_strawberry)")
    ap.add_argument("--redo", help="comma-separated ids to re-run (drop old rows first)")
    ap.add_argument("--max-tokens", type=int,
                    help="override max_tokens for every selected prompt")
    ap.add_argument("--results", default=RESULTS_FILE,
                    help="results JSONL path (use a fresh file for capped-vs-uncapped comparisons)")
    args = ap.parse_args()

    spec = json.load(open(PROMPTS_FILE))
    prompts = spec["prompts"]

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # Resume: ids already recorded are skipped (unless --redo).
    done = set()
    if os.path.exists(args.results):
        for line in open(args.results):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if args.redo:
        for pid_ in [x.strip() for x in args.redo.split(",") if x.strip()]:
            done.discard(pid_)

    if args.smoke:
        targets = [p for p in prompts if p["id"] == "d_r_strawberry"]
    elif args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        targets = [p for p in prompts if p["id"] in want]
    elif args.suite:
        targets = [p for p in prompts if p["suite"].lower() == args.suite]
    else:
        ap.error("need --suite, --ids, or --smoke")

    # Meta snapshot (first run only).
    if not os.path.exists(META_FILE):
        meta = {"server": server_snapshot(), "sampling": spec["sampling"],
                "n_prompts_total": len(prompts)}
        json.dump(meta, open(META_FILE, "w"), indent=1)
        print(f"meta -> {META_FILE}")

    out_f = open(args.results, "a")
    for p in targets:
        if p["id"] in done:
            print(f"SKIP {p['id']} (already recorded)")
            continue
        tools = [TOOL_SCHEMAS[p["tool"]]] if p.get("tool") in TOOL_SCHEMAS else None
        mt = args.max_tokens or p["max_tokens"]
        print(f"RUN  {p['id']} ({p['label']}) [max {mt}] ...", flush=True)
        try:
            r = stream_chat(p["prompt"], tools=tools, max_tokens=mt)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        rec = {"id": p["id"], "suite": p["suite"], "category": p["category"],
               "label": p["label"], "max_tokens": mt,
               "expected_tool": p.get("expected_tool"), **r}
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if "error" in r:
            print(f"  ERROR {r['error']}")
        else:
            tc = r["tool_calls"]
            tc_s = f" tool={tc[0]['name']}({tc[0]['arguments'][:60]})" if tc else ""
            print(f"  {r['completion_tokens']} tok in {r['total_s']}s  "
                  f"gen={r['gen_tok_s']} t/s  pp={r['pp_tok_s']} t/s  "
                  f"finish={r['finish_reason']}{tc_s}")
    out_f.close()

    # Summary of what is recorded so far.
    rows = [json.loads(x) for x in open(args.results) if x.strip()]
    print(f"\n{len(rows)} rows in {args.results}")
    by = {}
    for row in rows:
        if "gen_tok_s" in row and row.get("gen_tok_s"):
            by.setdefault(row["suite"], []).append(row["gen_tok_s"])
    for suite, vals in sorted(by.items()):
        print(f"  suite {suite}: n={len(vals)}  avg gen {sum(vals)/len(vals):.2f} tok/s")


if __name__ == "__main__":
    main()