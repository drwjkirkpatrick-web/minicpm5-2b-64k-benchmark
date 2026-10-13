#!/usr/bin/env python3
"""Chained runner: suites A -> B -> C -> D, each via bench.py, detached.

Resume-safe: bench.py skips ids already in the results JSONL.
Progress: batch.log + row count in results/bench_minicpm5_20261012.jsonl.
"""
import os
import subprocess
import sys

HERE = os.path.expanduser("~/projects/minicpm5-2b-64k-benchmark")
RESULTS = os.path.join(HERE, "results", "bench_minicpm5_20261012.jsonl")

log = open(os.path.join(HERE, "batch.log"), "w")
for suite in ["a", "b", "c", "d"]:
    print(f"=== suite {suite.upper()} starting ===", flush=True)
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "bench.py"), "--suite", suite],
        stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
    )
    n = sum(1 for _ in open(RESULTS)) if os.path.exists(RESULTS) else 0
    print(f"=== suite {suite.upper()} done rc={r.returncode} rows={n} ===", flush=True)
print("=== ALL SUITES DONE ===", flush=True)