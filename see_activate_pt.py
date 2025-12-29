# 查看 happy_steering_activations.pt 文件的参数
import torch

pt_path = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations.pt"   # 改成你的路径

# pt_path = "steering_activations.pt"
x = torch.load(pt_path, map_location="cpu")["steering_activations"]

print("shape:", x.shape)
print("dtype:", x.dtype)
print("device:", x.device)

print("min/max:", x.min().item(), x.max().item())
print("mean/std:", x.mean().item(), x.std().item())
print("has_nan:", torch.isnan(x).any().item())
print("has_inf:", torch.isinf(x).any().item())