#!/usr/bin/env python3
"""Launch MiniCPM5-2B MAXED: 131072 ctx + DSpark draft, detached.

Target: Q4_K_M, KV q8_0 @131072 = 2.82 GB. Draft: DSpark Q8_0 (dflash arch,
5 layers), KV q8_0 = 167 MB, --spec-type draft-dspark, block 7, both on GPU.
--predict -1 (default) = unlimited output headroom.
"""
import os
import subprocess
import sys
import urllib.request
import json
import time

SRV = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
TARGET = os.path.expanduser("~/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf")
DRAFT = os.path.expanduser("~/models/minicpm5-2b/minicpm5-2b-dspark-Q4_K_M.gguf")
LOG = os.path.expanduser("~/projects/minicpm5-2b-64k-benchmark/server_131k.log")
LOCK = "/tmp/oorep_bench_8090.lock"

for f in (TARGET, DRAFT):
    if not os.path.exists(f):
        sys.exit(f"missing: {f}")

try:
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
except FileExistsError:
    try:
        pid = int(open(LOCK).read().strip())
        os.kill(pid, 0)
        sys.exit("lock held by live pid %d" % pid)
    except (ProcessLookupError, ValueError):
        os.unlink(LOCK)
        open(LOCK, "w").write(str(os.getpid()))

log = open(LOG, "w")
cmd = [
    SRV,
    "-m", TARGET, "--alias", "minicpm5-2b-131k",
    "-md", DRAFT,
    "--spec-type", "draft-dspark",
    "--spec-draft-n-max", "7",
    "-ngl", "99", "-ngld", "99",
    "-c", "131072",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "-ctkd", "q8_0", "-ctvd", "q8_0",
    "-b", "256", "-ub", "256",
    "-fa", "on", "--jinja",
    "--host", "127.0.0.1", "--port", "8090",
    "-t", "6",
    "--predict", "-1",
]
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("server pid:", p.pid)

t0 = time.time()
while time.time() - t0 < 300:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=3) as r:
            if r.status == 200:
                print("healthy after %.1fs" % (time.time() - t0))
                break
    except Exception:
        if p.poll() is not None:
            print("SERVER DIED, exit", p.returncode)
            print(open(LOG).read()[-3000:])
            sys.exit(1)
        time.sleep(2)
else:
    print("health timeout")
    sys.exit(1)

with urllib.request.urlopen("http://127.0.0.1:8090/props", timeout=10) as r:
    props = json.load(r)
print("model:", props.get("model_path"))
print("n_ctx:", props.get("n_ctx"))