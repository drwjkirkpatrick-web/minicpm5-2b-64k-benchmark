#!/usr/bin/env python3
"""Run the full A-D suite against the live MiniCPM5-2B server, detached.

Streams progress to batch.log; rows append to results/bench_minicpm5_20261012.jsonl
(resume-safe). Cap profile v3 (A 6500, B 24000, C 4096/1024, D 3072).
"""
import os
import subprocess
import sys

HERE = os.path.expanduser("~/projects/minicpm5-2b-64k-benchmark")
log = open(os.path.join(HERE, "batch.log"), "w")
p = subprocess.Popen(
    [sys.executable, os.path.join(HERE, "bench.py"), "--suite", "a"],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=HERE,
)
print("batch A pid:", p.pid)