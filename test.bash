# python - <<'PY'
# import torch
# p="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train.style_emb.pt"

# try:
#     obj = torch.load(p, map_location="cpu", weights_only=True)
# except TypeError:
#     obj = torch.load(p, map_location="cpu")

# print("type:", type(obj))
# if isinstance(obj, dict):
#     print("keys:", list(obj.keys()))
#     if "keys" in obj: print("len(keys):", len(obj["keys"]))
#     if "emb" in obj:  print("emb:", obj["emb"].shape, obj["emb"].dtype)
#     if "items" in obj: 
#         print("len(items):", len(obj["items"]))
#         print("items[0]:", obj["items"][0])
#     if "skipped" in obj:
#         print("len(skipped):", len(obj["skipped"]))
#         print("skipped[0:2]:", obj["skipped"][:2])
# PY
python - <<'PY'
import torch

path = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/debug_CE_and_Align/tts_epoch_10_step_1973/model.pt"   # 改成你的实际路径
obj = torch.load(path, map_location="cpu")

def get_state_dict(o):
    # 1) 直接就是 state_dict（值都是 tensor）
    if isinstance(o, dict) and all(hasattr(v, "shape") for v in o.values()):
        return o
    # 2) 常见 ckpt 包装格式
    if isinstance(o, dict):
        for k in ["state_dict", "model", "model_state_dict", "model_state", "net", "module"]:
            if k in o and isinstance(o[k], dict):
                return o[k]
    # 3) 直接存了整个 nn.Module
    if hasattr(o, "state_dict"):
        return o.state_dict()
    raise TypeError(f"Unknown checkpoint format: {type(o)}")

sd = get_state_dict(obj)
keys = list(sd.keys())

targets = ["align_text_proj", "align_logit_scale"]
hit = [k for k in keys if any(t in k for t in targets)]
print("Total keys:", len(keys))
print("Hit keys:", len(hit))
for k in hit[:50]:
    v = sd[k]
    shape = tuple(v.shape) if hasattr(v, "shape") else None
    print(f"{k}: {shape}, dtype={getattr(v,'dtype',None)}")
PY
