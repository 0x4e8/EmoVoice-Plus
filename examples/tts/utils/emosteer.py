# 添加代码
# examples/tts/utils/emosteer.py
import torch
from contextlib import contextmanager

def _is_target_block(m: torch.nn.Module) -> bool:
    # 最鲁棒：只看类名（避免导入路径不一致）
    return m.__class__.__name__ == "BasicTransformerBlock"

def iter_blocks(estimator: torch.nn.Module):
    return [m for m in estimator.modules() if _is_target_block(m)]

class ActivationRecorder:
    """记录每个 block 输入 hidden_states 的均值向量（按层累加）。"""
    def __init__(self, max_layers=None, every_n_calls=1):
        self.max_layers = max_layers
        self.every_n_calls = every_n_calls
        self.call = 0
        self.sum = []
        self.cnt = []

    def _hook(self, layer_idx: int):
        def fn(module, args, kwargs):
            self.call += 1
            if self.every_n_calls > 1 and (self.call % self.every_n_calls != 0):
                return
            hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                hs = args[0]
            if not torch.is_tensor(hs) or hs.ndim != 3:
                return
            # hs: [B, T, C] -> mean over B,T => [C]
            v = hs.detach().float().mean(dim=(0, 1)).cpu()

            while len(self.sum) <= layer_idx:
                self.sum.append(None); self.cnt.append(0)
            self.sum[layer_idx] = v if self.sum[layer_idx] is None else (self.sum[layer_idx] + v)
            self.cnt[layer_idx] += 1
        return fn

    @contextmanager
    def attach(self, estimator: torch.nn.Module):
        handles = []
        blocks = iter_blocks(estimator)
        if self.max_layers is not None:
            blocks = blocks[: self.max_layers]
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_pre_hook(self._hook(i), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def mean_vecs(self):
        vecs = []
        for s, c in zip(self.sum, self.cnt):
            vecs.append(None if (s is None or c == 0) else (s / c))
        return vecs

class EmoSteerer:
    """
    mode:
      - add: hs += alpha * v
      - erase: hs -= (hs·v) v   (v需单位化)
      - replace: erase(src) 再 add(tgt)
    """
    def __init__(self, vecs, alpha=0.6, layers=None, mode="add", eps=1e-6,
                 steer_only_cond_branch=True):
        self.vecs = vecs
        self.alpha = float(alpha)
        self.layers = set(layers) if layers is not None else None
        self.mode = mode
        self.eps = eps
        self.steer_only_cond_branch = steer_only_cond_branch

    def _get_v(self, idx, device, dtype, dim):
        if idx >= len(self.vecs) or self.vecs[idx] is None:
            return None
        v = self.vecs[idx].to(device=device, dtype=dtype)
        if v.numel() != dim:
            return None
        v = v / (v.norm(p=2) + self.eps)
        return v

    def _apply(self, hs, v):
        # hs [B,T,C], v [C]
        if self.mode == "add":
            return hs + self.alpha * v.view(1, 1, -1)
        if self.mode == "erase":
            proj = (hs * v.view(1, 1, -1)).sum(dim=-1, keepdim=True)  # [B,T,1]
            return hs - proj * v.view(1, 1, -1)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _hook(self, layer_idx: int):
        def fn(module, args, kwargs):
            hs = kwargs.get("hidden_states", None)
            use_kwargs = True
            if hs is None and len(args) > 0:
                hs = args[0]; use_kwargs = False
            if not torch.is_tensor(hs) or hs.ndim != 3:
                return

            if self.layers is not None and layer_idx not in self.layers:
                return

            B, T, C = hs.shape
            v = self._get_v(layer_idx, hs.device, hs.dtype, C)
            if v is None:
                return

            # 兼容某些 CFG/双分支 batch：默认只 steer 前半（条件分支）
            if self.steer_only_cond_branch and B % 2 == 0 and B >= 2:
                hs1, hs2 = hs[:B//2], hs[B//2:]
                hs1 = self._apply(hs1, v)
                hs_new = torch.cat([hs1, hs2], dim=0)
            else:
                hs_new = self._apply(hs, v)

            if use_kwargs:
                kwargs["hidden_states"] = hs_new
                return args, kwargs
            else:
                new_args = list(args); new_args[0] = hs_new
                return tuple(new_args), kwargs
        return fn

    @contextmanager
    def attach(self, estimator: torch.nn.Module):
        handles = []
        blocks = iter_blocks(estimator)
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_pre_hook(self._hook(i), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

def save_vecs(path, vecs, meta=None):
    obj = {"vecs": vecs, "meta": meta or {}}
    torch.save(obj, path)

def load_vecs(path):
    obj = torch.load(path, map_location="cpu")
    return obj["vecs"] if isinstance(obj, dict) and "vecs" in obj else obj
