#!/usr/bin/env python3
"""Launch MiniCPM5-2B Q4_K_M @ 64K q8_0 KV on port 8090, detached (setsid).

Matches the production Qwen3.5-2B launch discipline from ~/bin/llm-64k:
-ngl 99 -c 65536 -ctk/ctv q8_0 -b 512 -ub 512 -fa on --jinja -np 1.
"""
import os
import subprocess
import sys
import urllib.request
import json
import time

SRV = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL = os.path.expanduser("~/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf")
LOG = os.path.expanduser("~/projects/minicpm5-2b-64k-benchmark/server.log")
LOCK = "/tmp/oorep_bench_8090.lock"

if not os.path.exists(MODEL):
    sys.exit(f"model not found: {MODEL}")

# lock guard — one bench campaign on 8090 at a time
try:
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
except FileExistsError:
    try:
        pid = int(open(LOCK).read().strip())
        os.kill(pid, 0)
        sys.exit("lock held by live pid %d — refusing" % pid)
    except (ProcessLookupError, ValueError):
        os.unlink(LOCK)  # stale
        open(LOCK, "w").write(str(os.getpid()))

log = open(LOG, "w")
cmd = [
    SRV, "-m", MODEL, "--alias", "minicpm5-2b",
    "-ngl", "99", "-c", "65536", "-ctk", "q8_0", "-ctv", "q8_0",
    "-b", "512", "-ub", "512", "-fa", "on", "--jinja",
    "--host", "127.0.0.1", "--port", "8090", "-np", "1", "-t", "6",
]
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("server pid:", p.pid)

t0 = time.time()
while time.time() - t0 < 180:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=3) as r:
            if r.status == 200:
                print("healthy after %.1fs" % (time.time() - t0))
                with urllib.request.urlopen("http://127.0.0.1:8090/props", timeout=5) as pr:
                    props = json.load(pr)
                print("model:", props.get("model_path"))
                print("n_ctx:", props.get("n_ctx"))
                break
    except Exception:
        if p.poll() is not None:
            print("SERVER DIED, exit", p.returncode)
            print(open(LOG).read()[-2000:])
            sys.exit(1)
        time.sleep(2)
else:
    print("health timeout")
    sys.exit(1)