#!/usr/bin/env python3
# Fetch only the `language_model.mtp.*` tensors (the MTP head, 1.5 GB) of an MLX build by HTTP range requests into one
# safetensors file for `origami config set speculation <file>`. Usage: fetch-mtp-head.py OUT.safetensors [REPO_RESOLVE_URL/]
# Standard library only; the default source is a public MLX build that ships the head (pinned revision).
import json, struct, sys, urllib.request
repo = sys.argv[2] if len(sys.argv) > 2 else "https://huggingface.co/ARC4NUM/Qwen3.8-Flash-Next-Uncensored-MLX-Serve-4bit/resolve/9ebf9993b1eaec96aec938bf883601b51a90393b/"
out = sys.argv[1]
idx = json.load(urllib.request.urlopen(repo + "model.safetensors.index.json"))["weight_map"]
names = sorted(k for k in idx if ".mtp." in k)
def rng(f, a, b):
    r = urllib.request.Request(repo + f, headers={"Range": f"bytes={a}-{b-1}"})
    d = urllib.request.urlopen(r).read(); assert len(d) == b - a, (f, a, b, len(d)); return d
headers = {}
for f in sorted(set(idx[k] for k in names)):
    n = struct.unpack("<Q", rng(f, 0, 8))[0]
    headers[f] = (n, json.loads(rng(f, 8, 8 + n)))
meta, blobs, off = {}, [], 0
for k in names:
    n, h = headers[idx[k]]; t = h[k]; a, b = t["data_offsets"]
    data = rng(idx[k], 8 + n + a, 8 + n + b)
    meta[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + len(data)]}
    blobs.append(data); off += len(data); print(k, t["dtype"], t["shape"], flush=True)
hb = json.dumps(meta).encode(); hb += b" " * (-len(hb) % 8)
with open(out + ".partial", "wb") as w:
    w.write(struct.pack("<Q", len(hb))); w.write(hb); [w.write(x) for x in blobs]
import os; os.rename(out + ".partial", out); print("bytes", off)
