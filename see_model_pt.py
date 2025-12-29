#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect a PyTorch checkpoint (.pt) to see:
- What top-level keys it has
- Whether it contains a state_dict (and where)
- How many parameters are in it
- Key prefix statistics (e.g., model.layers.*, lm_head, embed_tokens, etc.)
- Whether it looks like a full-model checkpoint or "delta" checkpoint

Usage:
  python inspect_ckpt.py --ckpt /path/to/EmoVoice-PP.pt --topk 50
"""

import argparse
import os
from collections import Counter, defaultdict
import torch


def human_num(x: int) -> str:
    # readable number: 12345678 -> 12.35M
    if x >= 10**9:
        return f"{x/10**9:.2f}B"
    if x >= 10**6:
        return f"{x/10**6:.2f}M"
    if x >= 10**3:
        return f"{x/10**3:.2f}K"
    return str(x)


def find_state_dict(obj):
    """
    Try common patterns:
      - obj is already a state_dict (dict[str, Tensor])
      - obj["state_dict"], obj["model"], obj["model_state_dict"], obj["module"]
    Return (state_dict, where)
    """
    if isinstance(obj, dict):
        # Case 1: looks like state_dict itself
        # heuristic: many keys and values are tensors
        tensor_values = 0
        total = 0
        for k, v in obj.items():
            total += 1
            if torch.is_tensor(v):
                tensor_values += 1
        if total > 0 and tensor_values / total > 0.6 and total > 50:
            return obj, "<root> (already a state_dict)"

        # Case 2: common nested keys
        for key in ["state_dict", "model", "model_state_dict", "module", "net", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                sd, where = find_state_dict(obj[key])
                if sd is not None:
                    return sd, f"{key} -> {where}"

    return None, None


def prefix_stats(keys, depth=2):
    """
    Count prefixes like:
      depth=1: "model", "lm_head", "embed_tokens"
      depth=2: "model.layers", "model.embed_tokens", ...
    """
    c = Counter()
    for k in keys:
        parts = k.split(".")
        pref = ".".join(parts[:depth]) if len(parts) >= depth else k
        c[pref] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint (e.g., EmoVoice-PP.pt)")
    ap.add_argument("--topk", type=int, default=40, help="Show top-k prefix groups")
    ap.add_argument("--depth", type=int, default=2, help="Prefix depth for grouping (1~4 is typical)")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print("=" * 90)
    print(f"[1] Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Top-level keys ({len(obj)}): {list(obj.keys())[:30]}{' ...' if len(obj) > 30 else ''}")
    else:
        print("Top-level is not a dict; cannot list keys.")

    print("=" * 90)
    print("[2] Locating state_dict ...")
    sd, where = find_state_dict(obj)
    if sd is None:
        print("❌ Could not find a state_dict-like dict inside this checkpoint.")
        print("If this is a TorchScript or something else, you may need a different loader.")
        return

    print(f"✅ Found state_dict at: {where}")
    keys = list(sd.keys())
    print(f"state_dict keys: {len(keys)}")

    # Count total params and tensors
    total_params = 0
    total_tensors = 0
    dtype_counter = Counter()
    shape_examples = []

    for k, v in sd.items():
        if torch.is_tensor(v):
            total_tensors += 1
            total_params += v.numel()
            dtype_counter[str(v.dtype)] += 1
            if len(shape_examples) < 10:
                shape_examples.append((k, tuple(v.shape), str(v.dtype)))

    print(f"Tensor entries: {total_tensors}/{len(keys)}")
    print(f"Total parameters (numel): {total_params} ({human_num(total_params)})")
    print(f"Dtype breakdown: {dict(dtype_counter)}")
    print("Shape examples (first 10 tensors):")
    for k, shp, dt in shape_examples:
        print(f"  - {k:70s} {str(shp):20s} {dt}")

    print("=" * 90)
    print(f"[3] Prefix grouping stats (depth={args.depth})")
    pref = prefix_stats(keys, depth=args.depth)
    for p, cnt in pref.most_common(args.topk):
        print(f"{p:45s}  {cnt:8d}")

    print("=" * 90)
    print("[4] Heuristic: is this likely a full Qwen backbone checkpoint?")
    # Heuristics: presence of many transformer block weights
    # Qwen-style names often include: model.layers.N., layers.N., transformer.h.N., etc.
    patterns = [
        "model.layers.",
        "layers.",
        "transformer.h.",
        "decoder.layers.",
        "model.embed_tokens",
        "embed_tokens",
        "lm_head",
    ]
    hits = defaultdict(int)
    for k in keys:
        for pat in patterns:
            if k.startswith(pat):
                hits[pat] += 1

    for pat in patterns:
        print(f"prefix '{pat}': {hits[pat]} keys")

    # If there are thousands of model.layers.* keys, it's almost certainly full or near-full model.
    looks_full = (hits["model.layers."] > 500) or (hits["layers."] > 500) or (hits["transformer.h."] > 500)
    looks_delta = (len(keys) < 500) and (hits["lm_head"] > 0 or hits["embed_tokens"] > 0)

    if looks_full:
        print("\n✅ Likely FULL (or near-full) backbone weights are included (many transformer block keys).")
        print("   Meaning: it probably overwrites large parts of Qwen, not just a small adapter/head.")
    elif looks_delta:
        print("\n✅ Likely a SMALL/DELTA checkpoint (few keys), could be head/projector/adapter-only.")
        print("   Meaning: you must load base Qwen weights first, then apply this checkpoint.")
    else:
        print("\n⚠️ Unclear (mixed). It may include some backbone parts + extra modules, or naming differs.")

    print("=" * 90)
    print("[5] Optional: list a few 'model.layers.0' keys if present")
    sample = [k for k in keys if "layers.0" in k][:30]
    if sample:
        for k in sample:
            print("  ", k)
    else:
        print("No keys containing 'layers.0' found.")

    print("=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()



# python test.py \
#   --ckpt /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt/EmoVoice-PP.pt \
#   --depth 2 --topk 60
