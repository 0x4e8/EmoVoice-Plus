# 添加代码
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
precompute_style_emb.py

Precompute utterance-level style/emotion embeddings for EmoVoice-DB jsonl,
using FunASR AutoModel + emotion2vec(+).

Why this script:
- EmoVoice-DB jsonl has "target_wav" like "audio/surprised/xxx.wav" (relative path).
  If your jsonl is moved elsewhere (e.g., train_shuffled_5times.jsonl), joining with jsonl_dir may fail.
  -> Use --audio_root to resolve relative paths: audio_root/target_wav.

Key features:
- Reads jsonl and resolves wav paths via (--audio_root first) then (jsonl_dir).
- Supports wav.scp (Kaldi style) input for FunASR emotion2vec. Embeddings are saved as .npy in output_dir.  (see model cards)
- DDP-friendly: torchrun sharding; merges partials on rank0.
- Dedup by key (default True): helpful when your jsonl repeats same key multiple times (e.g., shuffled_5times).

Output .pt format:
{
  "version": 2,
  "model": <str>,
  "audio_key": <str>,
  "audio_root": <str|None>,
  "granularity": "utterance"|"frame",
  "keys": [key0, key1, ...],
  "emb": FloatTensor [N, D],
  "items": [{"idx": orig_line_idx, "key": key, "wav": abs_wav_path}, ...],
  "skipped": [{"idx":..., "key":..., "wav":..., "reason":...}, ...],
}

Usage (your case):
torchrun --nproc_per_node 2 --master_port 29509 \
  examples/tts/utils/precompute_style_emb.py \
  --jsonl /.../EmoVoice-DB/train_shuffled_5times.jsonl \
  --out_pt /.../EmoVoice-DB/train.style_emb.pt \
  --audio_key target_wav \
  --audio_root /.../EmoVoice/EmoVoice-DB \
  --model /.../ckpt/emotion2vec_plus_large \
  --chunk_size 512 --batch_size 64 --cleanup
"""

import os
import re
import json
import glob
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# DDP helpers
# -------------------------
def _dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world, local_rank


def _maybe_init_dist(backend: str) -> bool:
    rank, world, local_rank = _dist_info()
    if world <= 1:
        return False
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def _barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _destroy_process_group():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# -------------------------
# FunASR import
# -------------------------
def _try_import_funasr():
    try:
        from funasr import AutoModel  # type: ignore
        return AutoModel
    except Exception as e:
        raise RuntimeError(
            "Failed to import funasr.AutoModel. Please ensure funasr is installed.\n"
            "Hint: pip install -U funasr modelscope\n"
            f"Original error: {e}"
        )


# -------------------------
# IO helpers
# -------------------------
def _read_jsonl_lines(jsonl_path: str) -> List[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _safe_json_loads(line: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        return json.loads(line), None
    except Exception as e:
        return None, str(e)


def _resolve_key(d: Dict, fallback_idx: int) -> str:
    for k in ["key", "utt_id", "utt", "id", "uid"]:
        v = d.get(k, None)
        if isinstance(v, str) and v:
            return v
    return f"utt_{fallback_idx:08d}"


def _resolve_audio_field(d: Dict, audio_key: str) -> Optional[str]:
    """
    audio_key supports comma-separated fallbacks, e.g.:
      --audio_key target_wav,source_wav,wav_path
    """
    keys = [x.strip() for x in audio_key.split(",") if x.strip()]
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v:
            return v
    return None


def _normalize_path(p: str) -> str:
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p)


def _resolve_wav_path(wav_raw: str, jsonl_dir: str, audio_root: Optional[str]) -> Tuple[str, List[str]]:
    """
    Resolve a wav path with candidates:
    1) If absolute: itself
    2) If audio_root provided: audio_root/wav_raw (and special handling if wav_raw starts with "EmoVoice-DB/")
    3) jsonl_dir/wav_raw

    Returns: (chosen_or_last_candidate, candidates_tried)
    """
    wav_raw = _normalize_path(wav_raw)
    candidates: List[str] = []

    if os.path.isabs(wav_raw):
        candidates = [wav_raw]
        return wav_raw, candidates

    if audio_root:
        audio_root = _normalize_path(audio_root)
        candidates.append(os.path.join(audio_root, wav_raw))
        if wav_raw.startswith("EmoVoice-DB/"):
            candidates.append(os.path.join(audio_root, wav_raw[len("EmoVoice-DB/"):]))
    candidates.append(os.path.join(jsonl_dir, wav_raw))

    # choose first existing
    for c in candidates:
        c_norm = _normalize_path(c)
        if os.path.exists(c_norm):
            return c_norm, [_normalize_path(x) for x in candidates]
    return _normalize_path(candidates[-1]), [_normalize_path(x) for x in candidates]


def _write_wav_scp(items: List[Dict], scp_path: str):
    """
    wav.scp:
      <wav_name> <wav_path>
    """
    with open(scp_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(f"{it['scp_id']} {it['wav']}\n")


def _find_embedding_npy(out_dir: str, scp_id: str) -> Optional[str]:
    # Most emotion2vec(+)/FunASR variants save <scp_id>.npy under output_dir (sometimes nested).
    direct = os.path.join(out_dir, f"{scp_id}.npy")
    if os.path.exists(direct):
        return direct
    cand = glob.glob(os.path.join(out_dir, "**", f"{scp_id}.npy"), recursive=True)
    if cand:
        return cand[0]
    # fallback: partial match
    cand2 = glob.glob(os.path.join(out_dir, "**", f"{scp_id}*.npy"), recursive=True)
    if cand2:
        return cand2[0]
    return None


def _load_embedding(npy_path: str) -> torch.Tensor:
    arr = np.load(npy_path)
    # normalize to [D]
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=0)
    return torch.from_numpy(arr).float()


def _chunk_iter(lst: List, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield i, lst[i:i + chunk_size]


def _merge_pt_files(pt_paths: List[str], out_pt: str, base_obj: Optional[Dict] = None):
    """
    Merge multiple partial pt files.
    Keeps order by 'idx' (original line index).
    """
    items: List[Dict] = []
    embs: List[torch.Tensor] = []
    skipped: List[Dict] = []
    seen_keys = set()

    for p in pt_paths:
        obj = torch.load(p, map_location="cpu")
        for it, emb in zip(obj.get("items", []), obj.get("embeddings", [])):
            k = it["key"]
            if k in seen_keys:
                continue
            seen_keys.add(k)
            items.append(it)
            embs.append(emb.unsqueeze(0) if emb.ndim == 1 else emb[:1])
        skipped.extend(obj.get("skipped", []))

    if not embs:
        raise RuntimeError("No embeddings to merge.")

    emb_all = torch.cat(embs, dim=0)  # [N, D]

    # sort by original idx for determinism
    order = sorted(range(len(items)), key=lambda i: items[i].get("idx", i))
    items = [items[i] for i in order]
    emb_all = emb_all[torch.tensor(order, dtype=torch.long)]

    keys = [it["key"] for it in items]

    out = dict(base_obj) if base_obj else {}
    out.update({
        "version": 2,
        "keys": keys,
        "emb": emb_all,
        "items": items,
        "skipped": skipped,
    })
    torch.save(out, out_pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--out_pt", type=str, required=True)

    ap.add_argument("--audio_key", type=str, default="target_wav",
                    help="json field for wav path. Supports fallbacks: a,b,c")
    ap.add_argument("--audio_root", type=str, default=None,
                    help="Root dir to resolve relative wav paths (recommended).")

    ap.add_argument("--model", type=str, default="iic/emotion2vec_plus_large",
                    help="FunASR model id or local path.")
    ap.add_argument("--granularity", type=str, default="utterance", choices=["utterance", "frame"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=512)

    ap.add_argument("--tmp_dir", type=str, default="./_style_emb_tmp")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dedup_by_key", action="store_true", default=True)
    ap.add_argument("--no_dedup_by_key", action="store_true", help="Disable dedup_by_key (not recommended).")

    ap.add_argument("--debug_first_n", type=int, default=8,
                    help="Print first N skipped examples for debugging.")
    ap.add_argument("--dist_backend", type=str, default="nccl")

    args = ap.parse_args()
    if args.no_dedup_by_key:
        args.dedup_by_key = False

    rank, world, local_rank = _dist_info()
    dist_on = _maybe_init_dist(args.dist_backend)

    jsonl_dir = os.path.dirname(_normalize_path(args.jsonl))
    audio_root = _normalize_path(args.audio_root) if args.audio_root else None

    # Resume: load existing keys on rank0 then broadcast
    computed_keys = set()
    existing_obj = None
    if args.resume and os.path.exists(args.out_pt) and rank == 0:
        try:
            existing_obj = torch.load(args.out_pt, map_location="cpu")
            for k in existing_obj.get("keys", []):
                computed_keys.add(k)
            print(f"[rank0] Resume: found {len(computed_keys)} keys in {args.out_pt}", flush=True)
        except Exception as e:
            print(f"[rank0] Resume load failed: {e}", flush=True)
            existing_obj = None

    if dist_on:
        obj = [list(computed_keys)] if rank == 0 else [[]]
        torch.distributed.broadcast_object_list(obj, src=0)
        computed_keys = set(obj[0])

    # Read & parse jsonl
    lines = _read_jsonl_lines(args.jsonl)

    stats = {
        "total_lines": len(lines),
        "empty_lines": 0,
        "bad_json": 0,
        "missing_audio_field": 0,
        "wav_not_found": 0,
        "duplicate_key": 0,
        "resume_skipped": 0,
        "usable": 0,
    }
    skipped_debug: List[Dict] = []

    seen_keys = set()

    items_all: List[Dict] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            stats["empty_lines"] += 1
            continue

        d, err = _safe_json_loads(line)
        if d is None:
            stats["bad_json"] += 1
            if len(skipped_debug) < args.debug_first_n:
                skipped_debug.append({"idx": idx, "key": None, "wav": None, "reason": f"bad_json: {err}", "line_head": line[:140]})
            continue

        key = _resolve_key(d, idx)
        if key in computed_keys:
            stats["resume_skipped"] += 1
            continue

        if args.dedup_by_key:
            if key in seen_keys:
                stats["duplicate_key"] += 1
                continue
            seen_keys.add(key)

        wav_raw = _resolve_audio_field(d, args.audio_key)
        if not wav_raw:
            stats["missing_audio_field"] += 1
            if len(skipped_debug) < args.debug_first_n:
                skipped_debug.append({"idx": idx, "key": key, "wav": None, "reason": f"missing audio_key={args.audio_key}"})
            continue

        wav, cand = _resolve_wav_path(wav_raw, jsonl_dir, audio_root)
        if not os.path.exists(wav):
            stats["wav_not_found"] += 1
            if len(skipped_debug) < args.debug_first_n:
                skipped_debug.append({"idx": idx, "key": key, "wav": wav_raw, "resolved": wav, "candidates": cand, "reason": "wav not found"})
            continue

        items_all.append({"idx": idx, "key": key, "wav": wav})
        stats["usable"] += 1

    if rank == 0:
        print(
            f"[rank0] Parsed {stats['total_lines']} lines | usable={stats['usable']} | "
            f"bad_json={stats['bad_json']} | missing_audio_field={stats['missing_audio_field']} | "
            f"wav_not_found={stats['wav_not_found']} | dup_key={stats['duplicate_key']} | "
            f"resume_skipped={stats['resume_skipped']}",
            flush=True
        )
        if stats["usable"] == 0:
            print("[rank0] First skipped examples:", flush=True)
            for x in skipped_debug:
                print("  ", x, flush=True)

    # Shard by rank (deterministic)
    items = [it for i, it in enumerate(items_all) if (i % world) == rank]
    print(f"[rank{rank}] Shard items: {len(items)} / {len(items_all)}", flush=True)

    # If nothing to do, still cleanly exit
    if len(items) == 0:
        _barrier()
        _destroy_process_group()
        return

    # Init FunASR emotion2vec model
    AutoModel = _try_import_funasr()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Some funasr versions accept device arg, some don't; be tolerant.
    try:
        emo_model = AutoModel(model=args.model, device=device)
    except TypeError:
        emo_model = AutoModel(model=args.model)

    # Temp dirs
    base_tmp = os.path.join(_normalize_path(args.tmp_dir), f"rank{rank}")
    os.makedirs(base_tmp, exist_ok=True)

    collected_items: List[Dict] = []
    collected_embs: List[torch.Tensor] = []
    local_skipped: List[Dict] = []

    # Process in chunks
    for chunk_start, chunk in _chunk_iter(items, args.chunk_size):
        chunk_dir = os.path.join(base_tmp, f"chunk_{chunk_start:09d}")
        os.makedirs(chunk_dir, exist_ok=True)

        # Build wav.scp (scp_id must be a safe token)
        chunk_items: List[Dict] = []
        for it in chunk:
            scp_id = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", it["key"])
            chunk_items.append({**it, "scp_id": scp_id})

        scp_path = os.path.join(chunk_dir, "wav.scp")
        _write_wav_scp(chunk_items, scp_path)

        # Run extraction.
        # emotion2vec model cards show callable usage: model(input=..., output_dir=..., granularity=..., extract_embedding=True)
        # plus_seed shows generate() usage; we support both.  (see cited model cards)
        gen_kwargs = dict(output_dir=chunk_dir, granularity=args.granularity, extract_embedding=True, batch_size=args.batch_size)
        try:
            if hasattr(emo_model, "generate"):
                emo_model.generate(scp_path, **gen_kwargs)
            else:
                emo_model(input=scp_path, **gen_kwargs)
        except TypeError:
            # fallback: some versions use "input" named arg in generate
            try:
                emo_model.generate(input=scp_path, **gen_kwargs)
            except Exception as e:
                raise RuntimeError(f"FunASR generate failed: {e}")

        # Collect embeddings
        for it in chunk_items:
            npy = _find_embedding_npy(chunk_dir, it["scp_id"])
            if not npy:
                local_skipped.append({"idx": it["idx"], "key": it["key"], "wav": it["wav"], "reason": "embedding npy not found"})
                continue
            try:
                emb = _load_embedding(npy)
                collected_items.append({"idx": it["idx"], "key": it["key"], "wav": it["wav"]})
                collected_embs.append(emb)
            except Exception as e:
                local_skipped.append({"idx": it["idx"], "key": it["key"], "wav": it["wav"], "reason": f"load npy failed: {e}"})

        if args.cleanup:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    if len(collected_embs) == 0:
        if rank == 0:
            print("[rank0] ERROR: No embeddings collected. Check --audio_root, wav paths, and FunASR output.", flush=True)
        _barrier()
        _destroy_process_group()
        raise RuntimeError(f"[rank{rank}] No embeddings collected.")

    embs = torch.stack(collected_embs, dim=0)  # [N, D]
    if args.fp16:
        embs = embs.half()

    part_out = args.out_pt + f".rank{rank}.pt"
    torch.save(
        {
            "version": 2,
            "model": args.model,
            "audio_key": args.audio_key,
            "audio_root": args.audio_root,
            "granularity": args.granularity,
            "items": collected_items,
            "embeddings": embs.cpu(),
            "skipped": local_skipped,
        },
        part_out
    )
    print(f"[rank{rank}] Saved partial: {part_out} items={len(collected_items)} skipped={len(local_skipped)}", flush=True)

    _barrier()

    # Merge on rank0
    if (not dist_on) or rank == 0:
        pt_paths = [args.out_pt + f".rank{r}.pt" for r in range(world)] if dist_on else [part_out]
        pt_paths = [p for p in pt_paths if os.path.exists(p)]
        if not pt_paths:
            _destroy_process_group()
            raise RuntimeError("No partial pt files found for merging.")

        # Merge new partials into a temp file first
        tmp_new = args.out_pt + ".tmp_new.pt"
        base_obj = {
            "model": args.model,
            "audio_key": args.audio_key,
            "audio_root": args.audio_root,
            "granularity": args.granularity,
        }
        _merge_pt_files(pt_paths, tmp_new, base_obj=base_obj)

        # If resume and old exists: merge old + new
        if args.resume and os.path.exists(args.out_pt) and existing_obj is not None:
            new_obj = torch.load(tmp_new, map_location="cpu")
            # build dict for fast merge
            old_keys = existing_obj.get("keys", [])
            old_emb = existing_obj.get("emb", None)
            old_items = existing_obj.get("items", [])
            if old_emb is None:
                # fallback: maybe old file stored embeddings under another name
                old_emb = existing_obj.get("embeddings", None)

            old_map = {}
            if old_emb is not None and len(old_keys) == old_emb.shape[0]:
                for k, e, it in zip(old_keys, old_emb, old_items):
                    old_map[k] = (e, it)

            # add new
            merged_map = dict(old_map)
            for k, e, it in zip(new_obj["keys"], new_obj["emb"], new_obj["items"]):
                merged_map[k] = (e, it)

            # sort by original idx if available (prefer old idx)
            merged_items = []
            merged_embs = []
            for k, (e, it) in merged_map.items():
                merged_items.append(it)
                merged_embs.append(e.unsqueeze(0) if e.ndim == 1 else e[:1])
            emb_all = torch.cat(merged_embs, dim=0)

            order = sorted(range(len(merged_items)), key=lambda i: merged_items[i].get("idx", i))
            merged_items = [merged_items[i] for i in order]
            emb_all = emb_all[torch.tensor(order, dtype=torch.long)]
            merged_keys = [it["key"] for it in merged_items]

            final_obj = dict(base_obj)
            final_obj.update({
                "version": 2,
                "keys": merged_keys,
                "emb": emb_all,
                "items": merged_items,
                "skipped": (existing_obj.get("skipped", []) + new_obj.get("skipped", [])),
            })
            torch.save(final_obj, args.out_pt)
            os.remove(tmp_new)
            print(f"[rank0] Resume-merged -> {args.out_pt}  keys={len(merged_keys)}", flush=True)
        else:
            # no resume merge: just move tmp_new to out_pt
            os.replace(tmp_new, args.out_pt)
            print(f"[rank0] Merged -> {args.out_pt}", flush=True)

    _barrier()
    _destroy_process_group()


if __name__ == "__main__":
    main()
