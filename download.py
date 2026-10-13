#!/usr/bin/env python3
"""Download MiniCPM5-2B Q4_K_M GGUF from openbmb/MiniCPM5-2B-GGUF."""
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="openbmb/MiniCPM5-2B-GGUF",
    filename="MiniCPM5-2B-Q4_K_M.gguf",
    local_dir="/home/walker/models/minicpm5-2b",
)
print("DONE:", path)