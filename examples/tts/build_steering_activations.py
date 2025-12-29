# 添加代码
# import os, json, math, argparse, inspect
# import torch
# import torchaudio

# # 复用你仓库里的 CosyVoice 初始化逻辑

# from utils.codec_utils import setup_codec
# from tts_config import TrainConfig, ModelConfig


# def load_wav_16k(path: str) -> torch.Tensor:
#     wav, sr = torchaudio.load(path)
#     if wav.dim() == 2 and wav.size(0) > 1:
#         wav = wav.mean(dim=0, keepdim=True)
#     if sr != 16000:
#         wav = torchaudio.transforms.Resample(sr, 16000)(wav)
#     return wav  # [1, T]


# # def extract_prompt_condition(codec_decoder, prompt_wav_16k: torch.Tensor, cosyvoice_version: int):
# #     fe = codec_decoder.frontend
# #     prompt_token, prompt_token_len = fe._extract_speech_token(prompt_wav_16k)
# #     if cosyvoice_version == 1:
# #         prompt_resamp = torchaudio.transforms.Resample(16000, 22050)(prompt_wav_16k)
# #     else:
# #         prompt_resamp = torchaudio.transforms.Resample(16000, 24000)(prompt_wav_16k)
# #     prompt_feat, prompt_feat_len = fe._extract_speech_feat(prompt_resamp)
# #     spk_emb = fe._extract_spk_embedding(prompt_wav_16k)
# #     return prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, spk_emb


# # def extract_prompt_condition(codec_decoder, prompt_wav_16k: torch.Tensor, cosyvoice_version: int):
# #     """
# #     prompt_wav_16k: [1, T] or [T], 16kHz waveform.
# #     Return tensors are on codec_decoder.frontend.device (通常是 CUDA)。
# #     """
# #     # ---- 关键修复：强制 CPU resample，避免 kernel/device mismatch ----
# #     prompt_wav_16k = prompt_wav_16k.detach()
# #     if prompt_wav_16k.dim() == 1:
# #         prompt_wav_16k = prompt_wav_16k.unsqueeze(0)
# #     prompt_wav_16k = prompt_wav_16k.cpu().float().contiguous()

# #     # speech token / spk embedding 用 16k
# #     flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_wav_16k)
# #     flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_wav_16k)

# #     # speech feat：v1 用 22.05k；v2 用 24k（和你 codec_utils.py 一致）
# #     if cosyvoice_version == 1:
# #         prompt_resamp = torchaudio.functional.resample(prompt_wav_16k, 16000, 22050)
# #     elif cosyvoice_version == 2:
# #         prompt_resamp = torchaudio.functional.resample(prompt_wav_16k, 16000, 24000)
# #     else:
# #         raise NotImplementedError

# #     prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_resamp)

# #     return flow_prompt_speech_token, flow_prompt_speech_token_len, prompt_speech_feat, prompt_speech_feat_len, flow_embedding

# def extract_prompt_condition(codec_decoder, wav16, cosyvoice_version: int):
#     # wav16: torch.Tensor [1, T] at 16k
#     wav16 = wav16.detach().cpu().float()  # ✅ 关键：CPU

#     frontend = codec_decoder.frontend

#     # prompt token (always from 16k)
#     ptok, _ = frontend._extract_speech_token(wav16)

#     # prompt feat depends on cosyvoice version
#     if cosyvoice_version == 1:
#         prompt_resamp = torchaudio.transforms.Resample(16000, 22050)(wav16)  # CPU ok
#         pfeat, _ = frontend._extract_speech_feat(prompt_resamp)
#     else:
#         prompt_resamp = torchaudio.transforms.Resample(16000, 24000)(wav16)
#         pfeat, _ = frontend._extract_speech_feat(prompt_resamp)

#     emb = frontend._extract_spk_embedding(wav16)
#     return ptok, pfeat, emb


# def call_flow_inference(flow, token, prompt_token, prompt_feat, spk_emb):
#     # 兼容 v1/v2 的 flow.inference 签名差异
#     sig = inspect.signature(flow.inference)
#     kwargs = dict(
#         token=token,
#         token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=token.device),
#         prompt_token=prompt_token,
#         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=token.device),
#         prompt_feat=prompt_feat,
#         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=token.device),
#         embedding=spk_emb,
#     )
#     if "flow_cache" in sig.parameters:
#         # v1 用 flow_cache
#         kwargs["flow_cache"] = torch.zeros(1, 80, 0, 2, device=token.device, dtype=prompt_feat.dtype)
#     if "finalize" in sig.parameters:
#         # v2 可能有 finalize
#         kwargs["finalize"] = True
#     _ = flow.inference(**kwargs)


# class ActivationCollector:
#     """
#     收集 ConditionalDecoder(estimator) 内所有 BasicTransformerBlock 的“第一残差流输入激活”（forward_pre_hook 的 args[0]），
#     并按 (group=block_idx//block_group, step=int(t*steps)) 统计均值。
#     """
#     def __init__(self, estimator, steps=32, block_group=5):
#         self.estimator = estimator
#         self.steps = int(steps)
#         self.block_group = int(block_group)

#         self.blocks = []
#         for _, m in estimator.named_modules():
#             if m.__class__.__name__ == "BasicTransformerBlock":
#                 self.blocks.append(m)

#         if len(self.blocks) == 0:
#             raise RuntimeError("在 estimator 里没有找到 BasicTransformerBlock，无法做 EmoSteer 激活统计。")

#         self.num_groups = math.ceil(len(self.blocks) / self.block_group)

#         self._handles = []
#         self._current_step = 0

#         self.sum = None   # [G, S, C] float64
#         self.cnt = torch.zeros(self.num_groups, self.steps, dtype=torch.long)

#         # 外层 hook：读 t（你这个 ConditionalDecoder.forward(x, mask, mu, t, ...) 的第 4 个参数）
#         self._handles.append(estimator.register_forward_pre_hook(self._outer_pre_hook))

#         # 内层 hook：抓每个 transformer block 的输入 hidden_states
#         for i, blk in enumerate(self.blocks):
#             self._handles.append(blk.register_forward_pre_hook(self._make_block_hook(i)))

#     def close(self):
#         for h in self._handles:
#             h.remove()
#         self._handles = []

#     def _outer_pre_hook(self, module, args):
#         # ConditionalDecoder.forward(x, mask, mu, t, ...)
#         t = None
#         if len(args) >= 4:
#             t = args[3]
#         # t 可能是标量/shape[B]
#         if isinstance(t, torch.Tensor):
#             tv = float(t.detach().flatten()[0].cpu().item())
#         elif t is None:
#             tv = 0.0
#         else:
#             tv = float(t)
#         step = int(tv * self.steps)
#         if step < 0:
#             step = 0
#         if step >= self.steps:
#             step = self.steps - 1
#         self._current_step = step

#     def _pool_to_vec(self, x: torch.Tensor) -> torch.Tensor:
#         # 期望 x 为 [B, L, C] 或 [B, C, L]
#         if x.dim() != 3:
#             x = x.view(x.size(0), -1, x.size(-1))
#         if x.shape[-1] <= 8 and x.shape[1] > x.shape[-1]:
#             # 很少见：如果最后维特别小，猜测是 [B,C,L]，交换
#             x = x.transpose(1, 2)

#         # 统一为 [B, L, C]
#         if x.shape[1] == 256 and x.shape[-1] != 256:
#             x = x.transpose(1, 2)

#         vec = x.mean(dim=(0, 1))  # [C]
#         return vec

#     def _make_block_hook(self, block_idx: int):
#         def hook(module, args):
#             x = args[0]
#             if not isinstance(x, torch.Tensor):
#                 return
#             vec = self._pool_to_vec(x).detach().to(torch.float64).cpu()  # [C]
#             g = block_idx // self.block_group
#             s = self._current_step

#             if self.sum is None:
#                 C = vec.numel()
#                 self.sum = torch.zeros(self.num_groups, self.steps, C, dtype=torch.float64)

#             self.sum[g, s] += vec
#             self.cnt[g, s] += 1
#         return hook

#     def mean(self) -> torch.Tensor:
#         if self.sum is None:
#             raise RuntimeError("没有收集到任何激活。")
#         denom = self.cnt.clamp_min(1).to(torch.float64).unsqueeze(-1)  # [G,S,1]
#         return (self.sum / denom)  # [G,S,C]


# @torch.no_grad()
# def collect_mean_for_list(codec_decoder, wav_list, cosyvoice_version, base_token, steps, block_group, device):
#     estimator = codec_decoder.model.flow.decoder.estimator
#     collector = ActivationCollector(estimator, steps=steps, block_group=block_group)

#     flow = codec_decoder.model.flow
#     for p in wav_list:
#         p = p.strip()
#         if not p:
#             continue
#         wav16 = load_wav_16k(p).to(device)
#         ptok, _, pfeat, _, emb = extract_prompt_condition(codec_decoder, wav16, cosyvoice_version)
#         ptok = ptok.to(device)
#         pfeat = pfeat.to(device)
#         emb = emb.to(device)

#         call_flow_inference(flow, base_token, ptok, pfeat, emb)

#     mean = collector.mean()
#     collector.close()
#     return mean  # [G,S,C]


# def read_list(path: str):
#     with open(path, "r", encoding="utf-8") as f:
#         return [ln.strip() for ln in f if ln.strip()]


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--codec_decoder_path", type=str, required=True)
#     ap.add_argument("--cosyvoice_version", type=int, default=1, choices=[1, 2])
#     ap.add_argument("--neutral_list", type=str, required=True)
#     ap.add_argument("--emotion_json", type=str, required=True,
#                     help='JSON: {"happy":"happy.txt","sad":"sad.txt",...} 每个 txt 里每行一个 wav 路径')
#     ap.add_argument("--base_token_wav", type=str, required=True)
#     ap.add_argument("--base_token_len", type=int, default=200)
#     ap.add_argument("--steps", type=int, default=32)
#     ap.add_argument("--block_group", type=int, default=5)
#     ap.add_argument("--out", type=str, default="steering_activations.pt")
#     args = ap.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 初始化 codec_decoder（CosyVoice / CosyVoice2）
#     train_cfg = TrainConfig(enable_ddp=False, enable_fsdp=False, enable_deepspeed=False)
#     model_cfg = ModelConfig(codec_decode=True,
#                             cosyvoice_version=args.cosyvoice_version,
#                             codec_decoder_path=args.codec_decoder_path)
#     codec_decoder = setup_codec(train_cfg, model_cfg)
#     codec_decoder.model.flow.to(device).eval()

#     # 固定 token 序列：从 base_token_wav 提取 speech tokens，截断到 base_token_len
#     base_wav16 = load_wav_16k(args.base_token_wav).to(device)
#     base_tok, _ = codec_decoder.frontend._extract_speech_token(base_wav16)
#     base_tok = base_tok[:, : args.base_token_len].to(device)  # [1, L]
#     if base_tok.numel() == 0:
#         raise RuntimeError("base_token_wav 提取 token 失败/过短。")

#     neutral_list = read_list(args.neutral_list)
#     emo_map = json.load(open(args.emotion_json, "r", encoding="utf-8"))

#     # 先算 neutral mean
#     neu_mean = collect_mean_for_list(
#         codec_decoder, neutral_list, args.cosyvoice_version,
#         base_tok, args.steps, args.block_group, device
#     )

#     out = {
#         "meta": {
#             "steps": args.steps,
#             "block_group": args.block_group,
#             "num_blocks": int(sum(1 for _, m in codec_decoder.model.flow.decoder.estimator.named_modules()
#                                   if m.__class__.__name__ == "BasicTransformerBlock")),
#         },
#         "neutral_mean": neu_mean,   # 可选：留着便于你调试/复现
#         "emotions": {}
#     }

#     # 每个 emotion：算 mean，再做差向量
#     for emo, lst_path in emo_map.items():
#         wavs = read_list(lst_path)
#         emo_mean = collect_mean_for_list(
#             codec_decoder, wavs, args.cosyvoice_version,
#             base_tok, args.steps, args.block_group, device
#         )
#         direction = (emo_mean - neu_mean).to(torch.float32)  # [G,S,C]
#         out["emotions"][emo] = {
#             "mean": emo_mean,
#             "direction": direction
#         }
#         print(f"[OK] emotion={emo} direction shape={tuple(direction.shape)}")

#     torch.save(out, args.out)
#     print(f"[DONE] saved to: {args.out}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build steering_activations.pt for EmoSteer-style activation steering on CosyVoice flow estimator.

Input JSON format (recommended):
{
  "neutral": ["abs/or/rel/path/to/neu_001.wav", "neu_002.wav", ...],
  "happy":   ["abs/or/rel/path/to/hap_001.wav", "hap_002.wav", ...],
  "sad":     [...],
  ...
}

This script:
- loads CosyVoice/CosyVoice2 codec decoder (flow + estimator)
- extracts prompt conditions from each reference wav (prompt_token, prompt_feat, speaker_embedding)
- runs flow.inference() with a fixed base token sequence to trigger estimator forward
- hooks BasicTransformerBlock to collect block input (or output fallback) as "activation"
- computes mean activations per layer for neutral and each emotion
- saves difference-in-means steering vectors (emotion - neutral) and normalized directions
"""

import os
import json
import math
import random
import inspect
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio


# -----------------------------
# Helpers: IO
# -----------------------------
def read_list_file(path: str) -> List[str]:
    """Read .txt list (one wav path per line)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            items.append(p)
    return items


def load_prompt_map(path: str) -> Dict[str, List[str]]:
    """
    Load prompt map from:
      - .json (dict emotion -> list[wav])
      - .txt (treated as neutral list only)
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("prompt_lists_json must be a dict emotion->list[wav]")
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.endswith(".txt"):
                out[k] = read_list_file(v)
            elif isinstance(v, list):
                out[k] = v
            else:
                raise ValueError(f"Invalid entry for '{k}': must be list[wav] or a .txt path")
        return out
    elif path.endswith(".txt"):
        return {"neutral": read_list_file(path)}
    else:
        raise ValueError("prompt_lists_json must be .json or .txt")


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def maybe_truncate_list(xs: List[str], max_n: int, seed: int) -> List[str]:
    if max_n <= 0 or len(xs) <= max_n:
        return xs
    rnd = random.Random(seed)
    ys = xs[:]
    rnd.shuffle(ys)
    return ys[:max_n]


# -----------------------------
# CosyVoice prompt condition extraction
# -----------------------------
@torch.no_grad()
def extract_prompt_condition(codec_decoder, wav16k: torch.Tensor, cosyvoice_version: int):
    """
    Returns:
      prompt_token (1, Ttok) int32 (on codec_decoder.frontend.device)
      prompt_feat  (1, Tfeat, 80) float (on codec_decoder.frontend.device)
      embedding    (1, D) float (on codec_decoder.frontend.device)
    NOTE: keep wav16k on CPU for resampling to avoid device mismatch issues.
    """
    assert wav16k.device.type == "cpu", "wav16k should stay on CPU to avoid torchaudio Resample device mismatch"

    # token + spk embedding extracted from 16k wav
    prompt_token, _ = codec_decoder.frontend._extract_speech_token(wav16k)
    embedding = codec_decoder.frontend._extract_spk_embedding(wav16k)

    # feat needs model SR: v1 uses 22050, v2 uses 24000
    if cosyvoice_version == 1:
        prompt_resamp = torchaudio.transforms.Resample(16000, 22050)(wav16k)
    elif cosyvoice_version == 2:
        prompt_resamp = torchaudio.transforms.Resample(16000, 24000)(wav16k)
    else:
        raise NotImplementedError(f"Unsupported cosyvoice_version={cosyvoice_version}")

    prompt_feat, _ = codec_decoder.frontend._extract_speech_feat(prompt_resamp)
    return prompt_token, prompt_feat, embedding


@torch.no_grad()
def extract_base_token(codec_decoder, base_wav16k: torch.Tensor, token_max_len: int) -> torch.Tensor:
    """
    Extract a fixed speech token sequence used as flow input token to trigger estimator forward.
    Return shape: (1, Ttok) int32 on codec_decoder.frontend.device.
    """
    assert base_wav16k.device.type == "cpu"
    base_token, _ = codec_decoder.frontend._extract_speech_token(base_wav16k)
    if token_max_len > 0 and base_token.shape[1] > token_max_len:
        base_token = base_token[:, :token_max_len]
    return base_token


# -----------------------------
# Hook collector
# -----------------------------
class ActivationCollector:
    """
    Collect mean activation vectors per BasicTransformerBlock.

    We try to use block input as activation (preferred for "residual stream" style).
    If the block is called with kwargs-only (args empty) and we can't access input,
    we fallback to using output in a forward_hook.
    """

    def __init__(self, layer_names: List[str], device: torch.device):
        self.layer_names = layer_names
        self.device = device

        # accumulators: label -> [num_layers, hidden_dim] sum and count
        self._sum: Dict[str, torch.Tensor] = {}
        self._cnt: Dict[str, torch.Tensor] = {}

        self.current_label: Optional[str] = None
        self.hidden_dim: Optional[int] = None

    def begin_label(self, label: str):
        self.current_label = label
        if label not in self._sum:
            # hidden_dim unknown until first hook fires; init later
            self._sum[label] = None
            self._cnt[label] = None

    def end_label(self):
        self.current_label = None

    def _init_label_buffers(self, label: str, hidden_dim: int):
        num_layers = len(self.layer_names)
        self._sum[label] = torch.zeros(num_layers, hidden_dim, device="cpu", dtype=torch.float32)
        self._cnt[label] = torch.zeros(num_layers, device="cpu", dtype=torch.long)

    def _reduce_to_vec(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert activation tensor x to a 1D vector [hidden_dim] by averaging batch/time dims.
        Supports shapes:
          [B, T, C]  -> mean over (0,1)
          [B, C, T]  -> mean over (0,2)
          [B, C]     -> mean over 0
        """
        if not torch.is_tensor(x):
            return None
        if x.numel() == 0:
            return None

        # detach, float32 on CPU to accumulate stably
        x = x.detach()

        if x.dim() == 3:
            # decide which dim is hidden
            # common cases: [B,T,C] or [B,C,T]
            if x.shape[-1] <= 4096 and x.shape[-1] >= 32:
                # assume last dim is hidden
                v = x.float().mean(dim=(0, 1))  # [C]
            elif x.shape[1] <= 4096 and x.shape[1] >= 32:
                # assume middle dim is hidden
                v = x.float().mean(dim=(0, 2))  # [C]
            else:
                # fallback: flatten last dim
                v = x.reshape(-1).float()
        elif x.dim() == 2:
            v = x.float().mean(dim=0)
        elif x.dim() == 1:
            v = x.float()
        else:
            v = x.reshape(-1).float()

        return v.cpu()

    def add(self, layer_idx: int, x: torch.Tensor):
        label = self.current_label
        if label is None:
            return
        v = self._reduce_to_vec(x)
        if v is None:
            return

        if self.hidden_dim is None:
            self.hidden_dim = int(v.numel())

        if self._sum[label] is None:
            self._init_label_buffers(label, self.hidden_dim)

        # if hidden_dim changes unexpectedly, skip
        if v.numel() != self._sum[label].shape[1]:
            return

        self._sum[label][layer_idx] += v
        self._cnt[label][layer_idx] += 1

    def get_mean(self, label: str) -> torch.Tensor:
        if label not in self._sum or self._sum[label] is None:
            raise RuntimeError(f"No activations collected for label={label}")
        s = self._sum[label]
        c = self._cnt[label].clamp_min(1).unsqueeze(1).float()
        return s / c  # [L, D]

    def get_counts(self, label: str) -> torch.Tensor:
        return self._cnt[label].clone() if label in self._cnt and self._cnt[label] is not None else None


class HookBundle:
    def __init__(self):
        self.handles = []

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def _supports_with_kwargs(register_fn) -> bool:
    try:
        sig = inspect.signature(register_fn)
        return "with_kwargs" in sig.parameters
    except Exception:
        return False


def attach_block_hooks(estimator: torch.nn.Module, collector: ActivationCollector) -> Tuple[HookBundle, List[str]]:
    """
    Find all BasicTransformerBlock modules under estimator and hook them.
    Returns: (bundle, layer_names)
    """
    blocks: List[Tuple[str, torch.nn.Module]] = []
    for name, m in estimator.named_modules():
        if m.__class__.__name__ == "BasicTransformerBlock":
            blocks.append((name, m))

    if len(blocks) == 0:
        raise RuntimeError("No BasicTransformerBlock found under estimator. Check your estimator architecture.")

    layer_names = [n for n, _ in blocks]
    collector.layer_names[:] = layer_names  # update in-place

    bundle = HookBundle()

    # Prefer forward_pre_hook to capture input, but handle kwargs-only calls safely.
    for layer_idx, (name, blk) in enumerate(blocks):

        def make_pre_hook(idx: int):
            def pre_hook(module, args, kwargs=None):
                # kwargs may be None depending on torch version/hook type
                x = None
                if args is not None and len(args) > 0:
                    x = args[0]
                else:
                    if kwargs:
                        # diffusers BasicTransformerBlock often uses "hidden_states"
                        if "hidden_states" in kwargs:
                            x = kwargs["hidden_states"]
                        elif "x" in kwargs:
                            x = kwargs["x"]
                if x is None:
                    return
                collector.add(idx, x)
            return pre_hook

        # If torch supports with_kwargs=True, use it (fixes your args-empty crash).
        if _supports_with_kwargs(blk.register_forward_pre_hook):
            h = blk.register_forward_pre_hook(make_pre_hook(layer_idx), with_kwargs=True)
            bundle.handles.append(h)
        else:
            # fallback: forward_hook (input may still be empty, but output exists)
            def make_fwd_hook(idx: int):
                def fwd_hook(module, args, out):
                    x = None
                    if args is not None and len(args) > 0:
                        x = args[0]
                    else:
                        x = out
                    if x is None:
                        return
                    collector.add(idx, x)
                return fwd_hook

            h = blk.register_forward_hook(make_fwd_hook(layer_idx))
            bundle.handles.append(h)

    return bundle, layer_names


# -----------------------------
# Call flow.inference to trigger estimator
# -----------------------------
@torch.no_grad()
def call_flow_inference(flow, token, prompt_token, prompt_feat, embedding):
    """
    Calls flow.inference with correct kwargs based on signature (v1/v2 differences).
    """
    device = next(flow.parameters()).device
    token = token.to(device)
    prompt_token = prompt_token.to(device)
    prompt_feat = prompt_feat.to(device)
    embedding = embedding.to(device)

    kwargs = dict(
        token=token,
        token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=device),
        prompt_token=prompt_token,
        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device),
        prompt_feat=prompt_feat,
        prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=device),
        embedding=embedding,
    )

    sig = inspect.signature(flow.inference)
    if "flow_cache" in sig.parameters:
        kwargs["flow_cache"] = torch.zeros(1, 80, 0, 2, device=device)
    if "finalize" in sig.parameters:
        kwargs["finalize"] = True

    _ = flow.inference(**kwargs)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_decoder_path", type=str, required=True,
                        help="CosyVoice/CosyVoice2 model dir or modelscope repo id (same as your model_config.codec_decoder_path)")
    parser.add_argument("--cosyvoice_version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--prompt_lists_json", type=str, required=True,
                        help="JSON dict emotion->list[wav] (must include 'neutral'). Each list item can be wav path.")
    parser.add_argument("--out_path", type=str, default="steering_activations.pt")
    parser.add_argument("--base_token_wav", type=str, default="",
                        help="A 16k wav used to extract base speech tokens. If empty, use the first neutral wav.")
    parser.add_argument("--token_max_len", type=int, default=200,
                        help="Max token length for base token sequence (reduce compute).")
    parser.add_argument("--max_refs_per_emotion", type=int, default=10,
                        help="Randomly subsample each emotion list to this size (<=0 means use all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:2")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load codec decoder using your existing helper (keeps behavior consistent with EmoVoice code) ---
    # NOTE: we import inside main so it works when run from examples/tts.
    from tts_config import TrainConfig, ModelConfig
    from utils.codec_utils import setup_codec

    train_config = TrainConfig()
    train_config.enable_ddp = False
    train_config.enable_fsdp = False

    model_config = ModelConfig()
    model_config.codec_decoder_path = args.codec_decoder_path
    model_config.codec_decoder_type = "CosyVoice"
    model_config.cosyvoice_version = int(args.cosyvoice_version)

    codec_decoder = setup_codec(train_config, model_config)

    # --- Resolve estimator & flow ---
    estimator = codec_decoder.model.flow.decoder.estimator
    flow = codec_decoder.model.flow

    # --- Load prompt lists ---
    prompt_map = load_prompt_map(args.prompt_lists_json)
    if "neutral" not in prompt_map:
        raise ValueError("prompt_lists_json must contain key 'neutral'")

    # subsample
    for k in list(prompt_map.keys()):
        prompt_map[k] = maybe_truncate_list(prompt_map[k], args.max_refs_per_emotion, args.seed + hash(k) % 10000)

    # pick base token wav
    if args.base_token_wav:
        base_wav_path = args.base_token_wav
    else:
        if len(prompt_map["neutral"]) == 0:
            raise ValueError("neutral list is empty, cannot auto-pick base_token_wav")
        base_wav_path = prompt_map["neutral"][0]

    ensure_exists(base_wav_path)

    # load wavs with CosyVoice load_wav (keeps amplitude conventions)
    from utils.cosyvoice.utils.file_utils import load_wav

    base_wav16k = load_wav(base_wav_path, 16000)  # CPU tensor
    base_token = extract_base_token(codec_decoder, base_wav16k, token_max_len=args.token_max_len)

    # --- Attach hooks ---
    collector = ActivationCollector(layer_names=[], device=torch.device(args.device))
    hook_bundle, layer_names = attach_block_hooks(estimator, collector)

    # --- Run collection ---
    def run_label(label: str, wav_list: List[str]):
        collector.begin_label(label)
        for p in wav_list:
            ensure_exists(p)
            wav16k = load_wav(p, 16000)  # keep on CPU (important)
            ptok, pfeat, emb = extract_prompt_condition(codec_decoder, wav16k, args.cosyvoice_version)
            call_flow_inference(flow, base_token, ptok, pfeat, emb)
        collector.end_label()

    # neutral first
    run_label("neutral", prompt_map["neutral"])

    # other emotions
    emotions = [k for k in prompt_map.keys() if k != "neutral"]
    for emo in emotions:
        run_label(emo, prompt_map[emo])

    # --- Compute directions ---
    neutral_mean = collector.get_mean("neutral")  # [L, D]
    results = {
        "meta": {
            "cosyvoice_version": int(args.cosyvoice_version),
            "codec_decoder_path": args.codec_decoder_path,
            "base_token_wav": base_wav_path,
            "token_max_len": int(args.token_max_len),
            "max_refs_per_emotion": int(args.max_refs_per_emotion),
            "seed": int(args.seed),
        },
        "layer_names": layer_names,
        "neutral_mean": neutral_mean,  # [L, D]
        "counts": {
            "neutral": collector.get_counts("neutral"),
        },
        "emotion_mean": {},
        "steer_vec": {},
        "steer_dir": {},
    }

    for emo in emotions:
        m = collector.get_mean(emo)  # [L, D]
        results["emotion_mean"][emo] = m
        results["counts"][emo] = collector.get_counts(emo)

        u = m - neutral_mean  # difference-in-means
        results["steer_vec"][emo] = u

        # normalize per-layer (unit direction)
        denom = torch.norm(u, dim=1, keepdim=True).clamp_min(1e-8)
        results["steer_dir"][emo] = u / denom

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(results, args.out_path)
    hook_bundle.remove()

    print(f"[OK] Saved steering activations to: {args.out_path}")
    print(f"     layers: {len(layer_names)} | hidden_dim: {neutral_mean.shape[1]}")
    print(f"     emotions: {emotions}")


if __name__ == "__main__":
    main()
