#!/usr/bin/env python3
"""Write grades JSON + compute final numbers for the MiniCPM5-2B report."""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(HERE, "results", "bench_minicpm5_20261012.jsonl")) if x.strip()]

GRADES = {
  "a_code": (8, "clean typed merge sort w/ docstrings; standard and correct"),
  "a_creative": (7, "lighthouse scene, vivid, coherent"),
  "a_iambic": (2, "tetrameter + heavy line repetition; not pentameter"),
  "a_math": (10, "complete sqrt2 contradiction proof, Q.E.D."),
  "a_prose": (8, "structured Hashimoto's pathophysiology, clinically plausible"),
  "b_html": (8, "complete self-contained page, embedded CSS, 1701 tok"),
  "b_python": (9, "exec_check PASS: runs clean, correct dept averages + top-3"),
  "b_c": (7, "pthread queue by inspection, plausible, not executed"),
  "b_basic": (3, "```trix fence typo, REM-heavy skeleton, incomplete game"),
  "b_julia": (7, "Newton-Raphson with docstring, standard"),
  "c_creative": (7, "attic-guitar story, warm and coherent"),
  "c_func_email": (8, "correctly drafts in text, no tool call (expected)"),
  "c_func_read_file": (10, "read_file called w/ correct path"),
  "c_func_sqlite": (0, "REFUSAL: 'unable to query databases' instead of terminal call"),
  "c_func_terminal": (10, "terminal called w/ df -h /"),
  "c_func_web_search": (10, "web_search called w/ sensible query"),
  "c_func_write_file": (9, "write_file called w/ path + content"),
  "c_html_game": (7, "complete click-the-target game"),
  "c_html_profiles": (7, "clean profile cards page"),
  "c_iambic": (1, "degenerate 'slow, slow, slow' repetition"),
  "c_math_proof": (9, "clean induction proof of sum formula"),
  "c_python": (9, "exec_check PASS: runs clean, avg 86.25, Diana/Charlie"),
  "d_batball": (10, "$0.05, clean derivation"),
  "d_binsearch": (8, "sorted precondition, mid/compare/halve"),
  "d_boxes": (4, "misassigns label logic, but reaches draw-anchors-symmetry idea"),
  "d_e_elephant": (1, "WRONG: says 3 e's (elephant has 2), hallucinated third"),
  "d_fib": (10, "exec_check PASS: memoized, fib(30) instant, verified programmatically"),
  "d_ice": (1, "REFUSAL: 'can't answer without context' on basic physics"),
  "d_piano": (1, "calls Fermi question a joke, answers zero"),
  "d_r_strawberry": (2, "WRONG: says 4 r's; enumerated a phantom 4th"),
  "d_sheep": (7, "correct 9, then hedges with second interpretations"),
  "d_sqrt2": (9, "valid infinite-descent variant"),
  "d_sum100": (10, "5050 via n(n+1)/2"),
  "d_switches": (3, "misses heat trick; confused two-bulb mapping"),
}

keyed = {r["id"]: r for r in rows}
out = []
for pid, (g, note) in GRADES.items():
    r = keyed[pid]
    out.append({"id": pid, "suite": r["suite"], "grade": g, "note": note,
                "completion_tokens": r.get("completion_tokens"), "gen_tok_s": r.get("gen_tok_s"),
                "finish_reason": r.get("finish_reason")})
json.dump(out, open(os.path.join(HERE, "results", "grades_minicpm5_20261012.json"), "w"), indent=1)

by = defaultdict(list)
for o in out:
    by[o["suite"]].append(o["grade"])
print("per-suite quality:", {s: round(sum(v) / len(v), 2) for s, v in by.items()})
all_g = [o["grade"] for o in out]
print("overall quality: %.2f/10  (n=%d)" % (sum(all_g) / len(all_g), len(all_g)))
gens = [r["gen_tok_s"] for r in rows if r.get("gen_tok_s")]
print("overall gen tok/s: %.2f  (min %.1f max %.1f)" % (sum(gens) / len(gens), min(gens), max(gens)))
pps = [r["pp_tok_s"] for r in rows if r.get("pp_tok_s")]
print("avg pp tok/s: %.1f" % (sum(pps) / len(pps)))
ttfts = [r["ttft_s"] for r in rows if r.get("ttft_s")]
print("avg ttft: %.2fs" % (sum(ttfts) / len(ttfts)))
traps = ["d_r_strawberry", "d_e_elephant", "d_batball", "d_sum100", "d_sheep", "d_sqrt2", "d_fib", "d_binsearch"]
trap_pass = sum(1 for t in traps if keyed[t].get("completion_tokens") and GRADES[t][0] >= 7)
print("hard traps >=7:", trap_pass, "/", len(traps))