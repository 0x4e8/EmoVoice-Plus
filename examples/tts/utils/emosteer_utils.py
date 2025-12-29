# 添加代码
# ================================================================================================================
# 版本-1
# 没有用
# import torch
# from contextlib import contextmanager

# def _to_int(x, default=0):
#     try:
#         if isinstance(x, torch.Tensor):
#             return int(x.item())
#         return int(x)
#     except Exception:
#         return default

# def _to_float(x, default=0.0):
#     try:
#         if isinstance(x, torch.Tensor):
#             return float(x.item())
#         return float(x)
#     except Exception:
#         return default

# def find_candidate_blocks(flow_model):
#     """
#     尽量不依赖具体类名：收集所有“看起来像 Transformer/DiT block”的模块。
#     你也可以把打印结果发我，我再帮你写成精准匹配版本。
#     """
#     blocks = []
#     for name, m in flow_model.named_modules():
#         cls = m.__class__.__name__.lower()
#         if any(k in cls for k in ["dit", "transformer", "block"]):
#             # 过滤太小的模块（可按需调整）
#             if hasattr(m, "forward"):
#                 blocks.append((name, m))
#     # 去重（同一个模块可能被多次命中）
#     uniq = []
#     seen = set()
#     for n, m in blocks:
#         if id(m) not in seen:
#             uniq.append((n, m))
#             seen.add(id(m))
#     return uniq

# def load_steering(path, device):
#     obj = torch.load(path, map_location="cpu")
#     if isinstance(obj, dict) and "steering_activations" in obj:
#         obj = obj["steering_activations"]
#     if not isinstance(obj, torch.Tensor):
#         raise ValueError("steering_path must load a Tensor or a dict containing 'steering_activations'")
#     return obj.to(device)

# def make_steering_hook(block_idx, emosteer_cfg, steering_acts, name=""):
#     """
#     参考 EmoSteer-TTS 附录的实现：forward_pre_hook 中解析 (x,t,time,...)，
#     用 time 映射 step，再对 first residual activations 做注入。:contentReference[oaicite:5]{index=5}
#     """
#     strength = float(emosteer_cfg.steering_strength)
#     num_steps = int(emosteer_cfg.num_steps)
#     block_group = int(emosteer_cfg.block_group)

#     def hook(module, inputs):
#         if not inputs:
#             return inputs

#         x = inputs[0]
#         if not isinstance(x, torch.Tensor) or x.dim() != 3:
#             return inputs  # 只处理 (B,L,C)

#         # 尽量兼容不同 flow 实现：time 通常在 inputs[2]（论文示例如此）:contentReference[oaicite:6]{index=6}
#         time = _to_float(inputs[2], default=0.0) if len(inputs) > 2 else 0.0
#         step = int(time * num_steps)
#         step = max(0, min(num_steps - 1, step))

#         # ref_audio_len（论文示例 inputs[7]）:contentReference[oaicite:7]{index=7}
#         ref_len = _to_int(inputs[7], default=x.size(1)) if len(inputs) > 7 else x.size(1)
#         ref_len = max(0, min(x.size(1), ref_len))

#         # 取对应 steering 向量
#         # steering_acts: [G, T, C] 或 [B, T, C]
#         group_idx = block_idx // block_group
#         group_idx = min(group_idx, steering_acts.size(0) - 1)
#         act = steering_acts[group_idx, step]  # [C]
#         if act.numel() != x.size(-1):
#             return inputs  # 维度不匹配就跳过，避免崩

#         # 构造只作用在前 ref_len token 的注入项
#         B, L, C = x.shape
#         act = act.view(1, 1, C).expand(B, ref_len, C)
#         if ref_len < L:
#             pad = torch.zeros((B, L - ref_len, C), device=x.device, dtype=x.dtype)
#             act = torch.cat([act, pad], dim=1)

#         # 规范化以减小分布漂移（论文 hook 里也会保持 norm）:contentReference[oaicite:8]{index=8}
#         with torch.no_grad():
#             x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
#             x_unit = x / x_norm
#             act_unit = act / torch.norm(act, dim=-1, keepdim=True).clamp_min(1e-6)
#             x_new = x_unit + strength * act_unit
#             x_new = x_new / torch.norm(x_new, dim=-1, keepdim=True).clamp_min(1e-6)
#             x_new = x_new * x_norm

#         new_inputs = (x_new,) + tuple(inputs[1:])
#         return new_inputs

#     return hook

# @contextmanager
# def emosteer_context(codec_decoder, decode_cfg, device):
#     """
#     在一次 token2wav 前挂 hook，结束后自动 remove。
#     """
#     em = getattr(decode_cfg, "emosteer", None) if decode_cfg is not None else None
#     if em is None or (not bool(getattr(em, "enable", False))):
#         yield
#         return

#     if not getattr(em, "steering_path", None):
#         raise ValueError("decode_config.emosteer.enable=true but steering_path is None")

#     flow = codec_decoder.model.flow
#     steering_acts = load_steering(em.steering_path, device=device)

#     blocks = find_candidate_blocks(flow)
#     handles = []
#     try:
#         for idx, (name, m) in enumerate(blocks):
#             if em.layers is not None and idx not in set(em.layers):
#                 continue
#             h = m.register_forward_pre_hook(make_steering_hook(idx, em, steering_acts, name=name))
#             handles.append(h)
#         yield
#     finally:
#         for h in handles:
#             h.remove()


# ================================================================================================================
# 版本-2
# 有用
# examples/tts/utils/emosteer_utils.py
# import torch
# from contextlib import contextmanager

# def load_steering(path, device):
#     """
#     兼容：
#     1) torch.save(tensor) -> Tensor
#     2) torch.save({"steering_activations": tensor})
#     3) 你之前生成的 dict（含 steer_dir/steer_vec 等）
#     """
#     obj = torch.load(path, map_location="cpu")

#     if isinstance(obj, dict):
#         if "steering_activations" in obj and isinstance(obj["steering_activations"], torch.Tensor):
#             obj = obj["steering_activations"]
#         elif "steer_dir" in obj and isinstance(obj["steer_dir"], torch.Tensor):
#             obj = obj["steer_dir"]
#         elif "steer_vec" in obj and isinstance(obj["steer_vec"], torch.Tensor):
#             obj = obj["steer_vec"]
#         else:
#             # 你也可以在这里按需继续加 key
#             raise ValueError(
#                 f"steering file is dict but no usable tensor key found. keys={list(obj.keys())}"
#             )

#     if not isinstance(obj, torch.Tensor):
#         raise ValueError("steering_path must load a Tensor or a dict containing a Tensor.")
#     return obj.to(device)

# def list_transformer_blocks(codec_decoder):
#     """
#     只取 flow.decoder.estimator 内部的 BasicTransformerBlock。
#     这样数量通常就是 64（你打印出来那种结构）。
#     """
#     est = codec_decoder.model.flow.decoder.estimator
#     blocks = []
#     for name, m in est.named_modules():
#         if m.__class__.__name__ == "BasicTransformerBlock":
#             blocks.append((name, m))
#     return est, blocks

# def _get_x_from_args_kwargs(args, kwargs):
#     # positional
#     if args and isinstance(args[0], torch.Tensor):
#         return args[0], "args"
#     # diffusers 风格：hidden_states=...
#     if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
#         return kwargs["hidden_states"], "kwargs_hidden_states"
#     # 有些实现用 x=...
#     if "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
#         return kwargs["x"], "kwargs_x"
#     return None, None

# def _set_x_back(args, kwargs, x_new, where):
#     if where == "args":
#         new_args = (x_new,) + tuple(args[1:])
#         return new_args, kwargs
#     kwargs = dict(kwargs)
#     if where == "kwargs_hidden_states":
#         kwargs["hidden_states"] = x_new
#         return args, kwargs
#     if where == "kwargs_x":
#         kwargs["x"] = x_new
#         return args, kwargs
#     return args, kwargs

# def make_steering_hook(block_idx, em_cfg, steering_acts, debug_name=""):
#     alpha = float(getattr(em_cfg, "steering_strength", 0.0))

#     def hook(module, args, kwargs):
#         # alpha=0 直接不改（避免你以为开了但其实强度是 0）
#         if alpha == 0.0:
#             return (args, kwargs)

#         x, where = _get_x_from_args_kwargs(args, kwargs)
#         if x is None or x.dim() != 3:
#             return (args, kwargs)

#         # 期望 x: (B, L, C)
#         B, L, C = x.shape

#         # --- 支持 2D: [num_blocks, C]（你现在的 happy_steering_activations.pt 就是这个） ---
#         if steering_acts.dim() == 2:
#             if block_idx >= steering_acts.size(0):
#                 return (args, kwargs)
#             v = steering_acts[block_idx]  # [C]
#             if v.numel() != C:
#                 return (args, kwargs)
#             v = v.view(1, 1, C).expand(B, L, C).to(device=x.device, dtype=x.dtype)

#         # --- 可选支持 3D: [G, T, C]（如果你以后真做了按 step 的 steering） ---
#         elif steering_acts.dim() == 3:
#             # 最简单：不依赖 time，就用第 0 step
#             g = min(block_idx, steering_acts.size(0) - 1)
#             v = steering_acts[g, 0]  # [C]
#             if v.numel() != C:
#                 return (args, kwargs)
#             v = v.view(1, 1, C).expand(B, L, C).to(device=x.device, dtype=x.dtype)
#         else:
#             return (args, kwargs)

#         # 按论文思路做“保持原 L2 norm 的 renorm”，更稳定:contentReference[oaicite:1]{index=1}
#         with torch.no_grad():
#             x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
#             x_unit = x / x_norm
#             v_unit = v / torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)

#             x_new = x_unit + alpha * v_unit
#             x_new = x_new / torch.norm(x_new, dim=-1, keepdim=True).clamp_min(1e-6)
#             x_new = x_new * x_norm

#         if getattr(em_cfg, "debug", False):
#             # 只打印一次也行，你也可以自己加计数器
#             print(f"[EmoSteer] block={block_idx} name={debug_name} alpha={alpha} "
#                   f"||delta||={float((x_new - x).abs().mean().item()):.6f}")

#         return _set_x_back(args, kwargs, x_new, where)

#     return hook

# @contextmanager
# def emosteer_context(codec_decoder, decode_cfg, device):
#     em = getattr(decode_cfg, "emosteer", None) if decode_cfg is not None else None
#     if em is None or (not bool(getattr(em, "enable", False))):
#         yield
#         return

#     if not getattr(em, "steering_path", None):
#         raise ValueError("decode_config.emosteer.enable=true but steering_path is None")

#     steering_acts = load_steering(em.steering_path, device=device)

#     _, blocks = list_transformer_blocks(codec_decoder)
#     if len(blocks) == 0:
#         raise RuntimeError("No BasicTransformerBlock found under flow.decoder.estimator. Cannot attach EmoSteer hooks.")

#     # 可选：做个形状 sanity check（你现在大概率会看到 steering_acts.shape[0]==len(blocks)==64）
#     if steering_acts.dim() == 2 and steering_acts.size(0) != len(blocks):
#         print(f"[EmoSteer][WARN] steering acts first dim={steering_acts.size(0)} "
#               f"!= num_blocks={len(blocks)}. Will clip by min().")

#     handles = []
#     try:
#         target = set(em.layers) if getattr(em, "layers", None) is not None else None

#         for idx, (name, m) in enumerate(blocks):
#             if target is not None and idx not in target:
#                 continue
#             h = m.register_forward_pre_hook(
#                 make_steering_hook(idx, em, steering_acts, debug_name=name),
#                 with_kwargs=True
#             )
#             handles.append(h)
#         yield
#     finally:
#         for h in handles:
#             h.remove()




# ================================================================================================================
# 版本-3
# 没有用
# examples/tts/utils/emosteer_utils.py
# import torch
# from contextlib import contextmanager

# def _to_bool(x, default=False):
#     try:
#         if isinstance(x, torch.Tensor):
#             return bool(x.item())
#         return bool(x)
#     except Exception:
#         return default

# def _to_int(x, default=0):
#     try:
#         if isinstance(x, torch.Tensor):
#             return int(x.item())
#         return int(x)
#     except Exception:
#         return default

# def _to_float(x, default=0.0):
#     try:
#         if isinstance(x, torch.Tensor):
#             return float(x.flatten()[0].item())
#         return float(x)
#     except Exception:
#         return default

# def _list_transformer_blocks(estimator):
#     """
#     只收集 BasicTransformerBlock，顺序按 named_modules 的出现顺序。
#     你 print(estimator) 的结构里，BasicTransformerBlock 才是最“像论文里 DiT block”的部分。
#     """
#     blocks = []
#     for name, m in estimator.named_modules():
#         if m.__class__.__name__ == "BasicTransformerBlock":
#             blocks.append((name, m))
#     return blocks

# def load_steering(path, device, num_steps=None):
#     """
#     兼容三种保存格式：
#       1) 直接 Tensor
#       2) dict['steering_activations']
#       3) 你之前 build 出来的 dict（含 steer_dir / steer_vec）
#     最终返回 Tensor: [G, T, C]
#     """
#     obj = torch.load(path, map_location="cpu")

#     if isinstance(obj, dict):
#         if "steering_activations" in obj:
#             obj = obj["steering_activations"]
#         elif "steer_dir" in obj:
#             obj = obj["steer_dir"]
#         elif "steer_vec" in obj:
#             obj = obj["steer_vec"]

#     if not isinstance(obj, torch.Tensor):
#         raise ValueError("steering_path must load a Tensor or a dict containing 'steering_activations'/'steer_dir'/'steer_vec'")

#     # 期望 [G,T,C]
#     if obj.dim() == 3:
#         return obj.to(device)

#     # 兼容你现在的 [64,256]：如果你确实是 (G*T, C) 扁平保存的，就 reshape 回来
#     if obj.dim() == 2:
#         if num_steps is None:
#             raise ValueError("2D steering tensor detected; please set decode_config.emosteer.num_steps so we can reshape to [G,T,C].")
#         GT, C = obj.shape
#         if GT % num_steps != 0:
#             raise ValueError(f"Cannot reshape steering tensor of shape {obj.shape} with num_steps={num_steps}.")
#         G = GT // num_steps
#         obj = obj.view(G, num_steps, C)
#         return obj.to(device)

#     raise ValueError(f"Unsupported steering tensor shape: {tuple(obj.shape)}")

# def make_steering_hook(block_idx, blocks_per_group, steering_acts, steering_strength):
#     """
#     尽量对齐论文实现：只在 conditional 分支 (drop_audio_cond=False) 注入，
#     并保持 x 的全局范数不变。:contentReference[oaicite:6]{index=6}
#     """
#     def hook(module, inputs):
#         # 论文里 inputs 形如：
#         # (x, t, time, mask, rope, drop_audio_cond, drop_text, ref_audio_len) :contentReference[oaicite:7]{index=7}
#         if not inputs or len(inputs) < 3:
#             return inputs

#         x = inputs[0]
#         if (not isinstance(x, torch.Tensor)) or x.dim() != 3:
#             return inputs

#         # 解析关键信息
#         time = _to_float(inputs[2], 0.0)
#         drop_audio_cond = _to_bool(inputs[5], False) if len(inputs) > 5 else False
#         ref_len = _to_int(inputs[7], x.size(1)) if len(inputs) > 7 else x.size(1)

#         # 只 steer conditional 分支（关键，否则 CFG 崩 -> 噪音）:contentReference[oaicite:8]{index=8}
#         if drop_audio_cond:
#             return inputs

#         B, L, C = x.shape
#         ref_len = max(0, min(L, ref_len))

#         # 根据 steering_acts 的维度推断 num_steps / group 索引
#         G, T, C2 = steering_acts.shape
#         if C2 != C:
#             return inputs  # 维度不匹配直接跳过

#         step = int(time * T)
#         step = max(0, min(T - 1, step))

#         group_idx = block_idx // max(1, blocks_per_group)
#         group_idx = max(0, min(G - 1, group_idx))

#         act = steering_acts[group_idx, step]  # [C]

#         # 只作用在前 ref_len token
#         if ref_len == 0:
#             return inputs
#         act = act.view(1, 1, C).expand(B, ref_len, C)
#         if ref_len < L:
#             pad = torch.zeros((B, L - ref_len, C), device=x.device, dtype=x.dtype)
#             act = torch.cat([act, pad], dim=1)

#         # 论文式：保持 x 的全局 norm 不变 :contentReference[oaicite:9]{index=9}
#         with torch.no_grad():
#             x_norm = torch.norm(x, dim=(1, 2), keepdim=True).clamp_min(1e-6)   # [B,1,1]
#             act_norm = torch.norm(act, dim=(1, 2), keepdim=True).clamp_min(1e-6)
#             act_unit = act / act_norm
#             x_new = x + float(steering_strength) * act_unit
#             x_new = x_new * (x_norm / torch.norm(x_new, dim=(1, 2), keepdim=True).clamp_min(1e-6))

#         return (x_new,) + tuple(inputs[1:])

#     return hook

# @contextmanager
# def emosteer_context(codec_decoder, decode_cfg, device):
#     em = getattr(decode_cfg, "emosteer", None) if decode_cfg is not None else None
#     if em is None or (not bool(getattr(em, "enable", False))):
#         yield
#         return

#     if not getattr(em, "steering_path", None):
#         raise ValueError("decode_config.emosteer.enable=true but steering_path is None")

#     # 只 hook estimator 内部的 transformer blocks
#     estimator = codec_decoder.model.flow.decoder.estimator
#     blocks = _list_transformer_blocks(estimator)
#     if len(blocks) == 0:
#         raise RuntimeError("No BasicTransformerBlock found in estimator; cannot attach EmoSteer hooks.")

#     steering_acts = load_steering(em.steering_path, device=device, num_steps=int(getattr(em, "num_steps", 32)))

#     # 关键：blocks_per_group 取 em.block_group（论文常用 block_idx//5）:contentReference[oaicite:10]{index=10}
#     blocks_per_group = int(getattr(em, "block_group", 5))
#     strength = float(getattr(em, "steering_strength", 0.0))

#     handles = []
#     try:
#         for idx, (name, m) in enumerate(blocks):
#             if getattr(em, "layers", None) is not None and idx not in set(em.layers):
#                 continue
#             h = m.register_forward_pre_hook(
#                 make_steering_hook(idx, blocks_per_group, steering_acts, strength)
#             )
#             handles.append(h)
#         yield
#     finally:
#         for h in handles:
#             h.remove()




# ================================================================================================================
# 版本-4
# 有用，效果和上一个似乎一样
# examples/tts/utils/emosteer_utils.py
import torch
from contextlib import contextmanager

def load_steering(path, device):
    """
    兼容：
    1) torch.save(tensor)
    2) torch.save({"steering_activations": tensor})
    3) 你 build 出来的 dict（含 steer_dir/steer_vec 等）
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "steering_activations" in obj and isinstance(obj["steering_activations"], torch.Tensor):
            obj = obj["steering_activations"]
        elif "steer_dir" in obj and isinstance(obj["steer_dir"], torch.Tensor):
            obj = obj["steer_dir"]
        elif "steer_vec" in obj and isinstance(obj["steer_vec"], torch.Tensor):
            obj = obj["steer_vec"]
        else:
            raise ValueError(f"steering file is dict but no usable tensor key found. keys={list(obj.keys())}")

    if not isinstance(obj, torch.Tensor):
        raise ValueError("steering_path must load a Tensor or a dict containing a Tensor.")
    return obj.to(device)

def list_transformer_blocks(codec_decoder):
    """
    只取 flow.decoder.estimator 内部的 BasicTransformerBlock。
    你打印的 estimator 结构中，BasicTransformerBlock 总数应为 64。
    """
    est = codec_decoder.model.flow.decoder.estimator
    # print("codec_decoder:\n", codec_decoder) # <cosyvoice.cli.cosyvoice.CosyVoice object at 0x14edd858e980>
    # print("codec_decoder.model:\n", codec_decoder.model) # <cosyvoice.cli.model.CosyVoiceModel object at 0x14ed85a8ece0>
    # print("codec_decoder.model.flow:\n", codec_decoder.model.flow) # MaskedDiffWithXvec()
    # print("codec_decoder.model.flow.decoder:\n", codec_decoder.model.flow.decoder) # ConditionalCFM()
    # print("codec_decoder.model.flow.decoder.estimator:\n", codec_decoder.model.flow.decoder.estimator) # ConditionalDecoder()
    blocks = []
    for name, m in est.named_modules():
        if m.__class__.__name__ == "BasicTransformerBlock":
            blocks.append((name, m))
    return est, blocks

def _get_hidden_states(args, kwargs):
    # positional
    if args and isinstance(args[0], torch.Tensor):
        return args[0], "args"
    # kwargs: hidden_states=...
    if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
        return kwargs["hidden_states"], "kwargs_hidden_states"
    # fallback
    if "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
        return kwargs["x"], "kwargs_x"
    return None, None

def _set_hidden_states(args, kwargs, x_new, where):
    if where == "args":
        new_args = (x_new,) + tuple(args[1:])
        return new_args, kwargs
    kwargs = dict(kwargs)
    if where == "kwargs_hidden_states":
        kwargs["hidden_states"] = x_new
        return args, kwargs
    if where == "kwargs_x":
        kwargs["x"] = x_new
        return args, kwargs
    return args, kwargs

def make_steering_hook(block_idx, em_cfg, steering_acts, debug_name=""):
    alpha = float(getattr(em_cfg, "steering_strength", 0.0))
    beta  = float(getattr(em_cfg, "erasing_strength", 0.0))

    def hook(module, args, kwargs):
        # 两个都为 0 就不改
        if alpha == 0.0 and beta == 0.0:
            return (args, kwargs)

        x, where = _get_hidden_states(args, kwargs)
        if x is None or x.dim() != 3:
            return (args, kwargs)

        B, L, C = x.shape

        # 你的 steering_acts = [64, 256]（每个 block 一个向量）
        if steering_acts.dim() == 2:
            if block_idx >= steering_acts.size(0):
                return (args, kwargs)
            v = steering_acts[block_idx]  # [C]
            if v.numel() != C:
                return (args, kwargs)
            v = v.to(device=x.device, dtype=x.dtype).view(1, 1, C).expand(B, L, C)
        else:
            # 你当前不是 step-wise steering，就先不支持 3D，避免误用导致噪音
            return (args, kwargs)

        # 单位化方向
        eps = 1e-6
        with torch.no_grad():
            x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
            x_unit = x / x_norm

            v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
            v_unit = v / v_norm

            x_new = x_unit

            # 1) 情感增强：x_unit + alpha * v_unit
            if alpha != 0.0:
                x_new = x_new + alpha * v_unit

            # 2) 情感“擦除”：去掉在 v_unit 方向上的投影（更稳定、比直接减向量不容易炸）
            #    x_new = x_new - beta * proj_{v}(x_new)
            if beta != 0.0:
                proj = (x_new * v_unit).sum(dim=-1, keepdim=True) * v_unit
                x_new = x_new - beta * proj

            # 归一回原来的 token-norm，保持幅度不乱飞
            x_new = x_new / torch.norm(x_new, dim=-1, keepdim=True).clamp_min(eps)
            x_new = x_new * x_norm

        if getattr(em_cfg, "debug", False):
            delta = float((x_new - x).abs().mean().item())
            print(f"[EmoSteer] block={block_idx} name={debug_name} alpha={alpha} beta={beta} mean|Δ|={delta:.6e}")

        return _set_hidden_states(args, kwargs, x_new, where)

    return hook

@contextmanager
def emosteer_context(codec_decoder, decode_cfg, device):
    em = getattr(decode_cfg, "emosteer", None) if decode_cfg is not None else None
    if em is None or (not bool(getattr(em, "enable", False))):
        yield
        return

    if not getattr(em, "steering_path", None):
        raise ValueError("decode_config.emosteer.enable=true but steering_path is None")

    steering_acts = load_steering(em.steering_path, device=device)

    _, blocks = list_transformer_blocks(codec_decoder)
    if len(blocks) == 0:
        raise RuntimeError("No BasicTransformerBlock found under flow.decoder.estimator.")

    if steering_acts.dim() == 2 and steering_acts.size(0) != len(blocks):
        print(f"[EmoSteer][WARN] steering acts first dim={steering_acts.size(0)} != num_blocks={len(blocks)}.")

    handles = []
    try:
        target = set(em.layers) if getattr(em, "layers", None) is not None else None
        for idx, (name, m) in enumerate(blocks):
            if target is not None and idx not in target:
                continue
            h = m.register_forward_pre_hook(
                make_steering_hook(idx, em, steering_acts, debug_name=name),
                with_kwargs=True,   # 关键：你这里 transformer_block 用的是 hidden_states=...（kwargs）
            )
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()
