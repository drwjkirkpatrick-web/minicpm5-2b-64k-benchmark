# MiniCPM5-2B — 64K Benchmark — 2026-10-12/13

**Model:** MiniCPM5-2B Q4_K_M (1.56 GB, LlamaForCausalLM arch) — native ctx **131,072**,
no YaRN needed at 64K. 42 layers, **2 KV heads** (8:1 GQA) → KV at 64K = 1,409 MB q8_0.
GGUF from `openbmb/MiniCPM5-2B-GGUF` (Apache 2.0).

**Hardware:** Jetson Orin Nano 8 GB, **GUI off (gdm3 stopped)**, gateway resident.
**Server:** llama.cpp llama-server, `-c 65536 -ctk q8_0 -ctv q8_0 -ngl 99 -b 512 -ub 512 -fa on --jinja -np 1 -t 6`.
**Load:** 6.0 s to healthy, RSS steady at **3.41 GB** (1.56 weights + 1.41 KV + buffers), min-free ~2.5 GB. Comfortable; f16 KV (2.8 GB) would also fit.
**Config note:** thinking is **ON by default** (streams to `reasoning_content`). All 34 prompts
run with `chat_template_kwargs {"enable_thinking": false}` — the same thinking-off
configuration as the Qwen3.5-2B graded run. With thinking ON, a trivial question took
63 tokens and the strawberry trap burned its entire 3072-token cap on visible
"but wait…" re-verification (loop observed once in smoke; see Findings).

**Method:** canonical A–D 34-prompt suite (`prompts_abcd.json`, verbatim), cap profile v3
(A 6500 · B 24000 · C 4096/1024 · D 3072), temp 0.3 / top_p 0.95 / top_k 40,
streaming tok/s, exec_check code executed not inspected. Live resident server,
resume-safe JSONL. Raw data: `results/bench_minicpm5_20261012.jsonl`.

## Speed (generation tok/s, streaming, no drafter)

| Suite | n | Avg gen tok/s | Range | Avg ct | Capped |
|---|---|---|---|---|---|
| A — general | 5 | 25.95 | 25.2–26.5 | 616 | 0/5 |
| B — one-shot coding | 5 | 25.59 | 25.1–26.1 | 1040 | 0/5 |
| C — zoo/ornith | 12 | 23.97 | 18.3–26.6 | 272 | 0/12 |
| D — reasoning traps | 12 | 26.30 | 25.7–27.2 | 341 | 0/12 |
| **Overall** | **34** | **25.32** | peak 27.2 | | **0/34** |

- Prompt processing: avg 431 tok/s · TTFT: 0.14 s
- Tool-call rows dip to 18–19 tok/s (short bursts, 17–46 tokens), same pattern as LFM.
- **Zero truncations** with thinking off.

## Quality (graded 1–10 against grade_key)

| Suite | Avg | Notes |
|---|---|---|
| A | 7.0 | sqrt2 proof 10/10; iambic pentameter 2/10 (tetrameter + repetition) |
| B | 6.8 | b_python exec PASS (correct averages/top-3); TRS-80 BASIC 3/10 (```trix fence, REM skeleton) |
| C | 7.25 | proofs solid (9/10); c_python exec PASS; **c_func_sqlite refusal = 0** |
| D | 5.5 | batball/sum100/sqrt2/fib/binsearch/sheep pass; **strawberry+elephant both wrong** |
| **Overall** | **6.53/10** | |

**exec_check (executed, not inspected):** b_python ✓ (dept averages 75k/90k, top-3 Eve/Bob/Charlie) · c_python ✓ (avg 86.25, Diana/Charlie) · d_fib ✓ (memoized; fib(30)=832040 in <0.1 s, verified programmatically).

**Function calls: 5/6.** web_search / terminal (df -h /) / read_file / write_file all called with correct names and sensible arguments; c_func_email correctly drafts in text with no call. **c_func_sqlite refused** ("unable to query databases or access patient information") instead of issuing the terminal call — a guardrail-style refusal on a benign agentic task.

**Hard traps: 6/12** (grade ≥7): batball ✓, sum100 ✓, sqrt2 ✓, fib ✓, binsearch ✓, sheep ✓.
Fails: **strawberry (says 4, not 3)** · **elephant (says 3, not 2 — hallucinated a third 'e')** · boxes (misassigns label logic, partial) · switches (misses the heat trick) · ice (**refused**: "can't answer without context") · piano (calls the Fermi question a joke, answers zero).

## Findings

1. **Letter-counting is its blind spot** — both r-in-strawberry and e-in-elephant wrong,
   with confident letter-by-letter enumeration that inserts phantom letters. Qwen3.5-2B
   passes both.
2. **Refusal pattern** — 2/34 prompts refused benign questions (sqlite function call,
   "why ice floats"). Smells like over-tuned safety; directly harms agentic reliability.
3. **Low-temperature repetition pathology, stochastic** — smoke run of strawberry looped
   "But wait…" to the 3072 cap (correct answer, 9 "wait"s); the recorded formal run didn't
   loop but miscounted. Both iambic prompts degenerated into repeated phrases. Watch for
   loops on long generations; repeat_penalty 1.0 was standard here, untested variants may fix.
4. **Thinking mode**: default-on costs 30–60× token overhead on trivial prompts; the
   embedded switch (`enable_thinking: false`) works cleanly through llama.cpp — out of the
   box this is a thinking model, not a chat model.
5. **Memory champion** — 3.41 GB RSS at 64K q8_0 KV is the leanest 64K fit tested on this
   box (LFM2.5-3B: 4.84 GB). Could run f16 KV at 64K (−quantization risk) in ~4.8 GB.

## Comparison (same suite, same caps, same sampling)

| Model | Config | Gen tok/s | Quality | Hard traps* | Function calls | RSS @64K |
|---|---|---|---|---|---|---|
| **Qwen3.5-2B** | Q4_K_M, q8_0 KV, MTP, thinking off | **37.0** | **7.4/10** | **4/4** | 6/6 first-turn | ~4.3 GB |
| **LFM2.5-VL-3B** | Q4_K_M + DSpark, f16 KV, YaRN 2.0 | 35.2 | — | traps 5/5 | 5/6 | 4.84 GB |
| **MiniCPM5-2B** | Q4_K_M, q8_0 KV, no drafter, thinking off | 25.3 | 6.5/10 | 3/4 (of Qwen's four) | 5/6 (1 refusal) | **3.41 GB** |

*Qwen's graded four: strawberry/bat-ball/sheep/sum100 — MiniCPM5 misses strawberry.

## Verdict

**Not a Qwen3.5-2B replacement for the resident Hermes-agent slot.** Slower without a
drafter (25 vs 37 tok/s), lower quality (6.5 vs 7.4), fails the letter traps Qwen nails,
and refuses benign tasks. Its genuine strengths: leanest memory profile tested at 64K,
clean tool-calling when engaged, solid math/proof/code fundamentals (all exec_checks pass,
proofs 9–10/10), and native 131K context. OpenBMB ships a **DSpark draft model** — the
obvious follow-up is a speed pass with the drafter active; MiniCPM5's small KV might also
make it the best candidate for a 96K/128K context experiment on this 8 GB box.

## Files

- `launch_server.py` — detached launch + health/props check
- `bench.py` — canonical harness (from qwen35-2b-benchmark, results paths localized)
- `run_all.py` — chained A→D runner (detached)
- `summarize.py` / `write_grades.py` — metrics + grades
- `results/` — JSONL, meta, per-prompt responses, exec/ code + verified outputs, grades JSON