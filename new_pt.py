# import math
# import numpy as np
# import torch

# # =========================
# # 你要的“写死的参数”
# # =========================
# SRC_PT = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations.pt"          # 可选：用来校验shape/dtype（不依赖也能生成）
# DST_PT = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations_random.pt"    # 输出文件名
# SEED = 2025

# SHAPE = (64, 256)          # 与你现有的一样
# DTYPE = torch.float32      # 与你现有的一样

# T_MIN = -0.4956580400466919
# T_MAX = 0.47732457518577576
# T_MEAN = 0.0014065077994018793
# T_STD = 0.06248607859015465
# # =========================


# def safe_load(path: str):
#     try:
#         return torch.load(path, map_location="cpu", weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location="cpu")


# def main():
#     # 可选校验：如果 SRC_PT 存在且格式正确，就核对一下 shape/dtype
#     try:
#         obj = safe_load(SRC_PT)
#         if isinstance(obj, dict) and "steering_activations" in obj:
#             x0 = obj["steering_activations"]
#             if tuple(x0.shape) != SHAPE:
#                 print(f"[WARN] SRC shape={tuple(x0.shape)} != SHAPE={SHAPE}，仍按 SHAPE 生成")
#             if x0.dtype != DTYPE:
#                 print(f"[WARN] SRC dtype={x0.dtype} != DTYPE={DTYPE}，仍按 DTYPE 生成")
#     except Exception as e:
#         print(f"[INFO] 未读取 SRC_PT（{e}），直接按 SHAPE/DTYPE 生成")

#     N = SHAPE[0] * SHAPE[1]
#     M = N - 2  # 除去 min/max 两个位置
#     assert M % 2 == 0, "这里为了方便用等量 +1/-1，要求 M 为偶数"

#     rng = np.random.default_rng(SEED)

#     # 1) 构造 z ∈ {+1,-1}，且 sum(z)=0，保证中间部分均值可控、且值域有界
#     z = np.ones(M, dtype=np.float64)
#     z[: M // 2] = -1.0
#     rng.shuffle(z)
#     sum_z2 = float(np.dot(z, z))  # = M

#     # 2) 先按“数学上精确”的公式求 beta/alpha（让均值/方差满足目标）
#     beta = (N * T_MEAN - T_MIN - T_MAX) / M
#     delta = beta - T_MEAN

#     D_target = (N - 1) * (T_STD ** 2)  # sum (x - mean)^2
#     D_fixed = (T_MIN - T_MEAN) ** 2 + (T_MAX - T_MEAN) ** 2 + M * (delta ** 2)
#     alpha2 = (D_target - D_fixed) / sum_z2
#     if alpha2 <= 0:
#         raise RuntimeError("无法构造：alpha^2 <= 0（目标统计量与约束不兼容）")
#     alpha = math.sqrt(alpha2)

#     mid = beta + alpha * z  # float64

#     # 3) 随机挑两个位置放 min/max，剩下填 mid
#     perm = rng.permutation(N)
#     idx_min = int(perm[0])
#     idx_max = int(perm[1])

#     flat = np.empty(N, dtype=np.float64)
#     flat[idx_min] = T_MIN
#     flat[idx_max] = T_MAX

#     mask = np.ones(N, dtype=bool)
#     mask[idx_min] = False
#     mask[idx_max] = False
#     flat[mask] = mid

#     x = torch.tensor(flat.reshape(SHAPE), dtype=DTYPE)

#     # 4) 用“两点微调”把 mean 在 float32 下“精确对齐”，同时保持 std 精确不变
#     #    （上面那一步 mean 通常已经极近，但为了你要求“打印出来一模一样”，这里做强制对齐）
#     x_flat = x.flatten()

#     # 找到一个“中间部分的最大值”和“中间部分的最小值”的位置（避开全局 min/max 那两个点）
#     mask_mid = torch.ones(N, dtype=torch.bool)
#     mask_mid[idx_min] = False
#     mask_mid[idx_max] = False

#     mid_vals = x_flat[mask_mid]
#     hi_val = mid_vals.max().item()
#     lo_val = mid_vals.min().item()

#     # 拿到 hi_val / lo_val 的全局索引
#     hi_idx = int((x_flat == hi_val).nonzero(as_tuple=False)[0].item())
#     # lo_idx 不能和 hi_idx 重复
#     lo_candidates = (x_flat == lo_val).nonzero(as_tuple=False).flatten().tolist()
#     lo_idx = int(lo_candidates[0] if lo_candidates[0] != hi_idx else lo_candidates[1])

#     mean_cur = x.mean().item()
#     DeltaS = N * (T_MEAN - mean_cur)  # 需要补的“总和差值”

#     # 以目标均值 T_MEAN 为中心的当前离差平方和（用 double 计算更稳）
#     D0 = ((x_flat.double() - T_MEAN) ** 2).sum().item()
#     DeltaQ = D_target - D0

#     xi = float(x_flat[hi_idx].item())
#     xj = float(x_flat[lo_idx].item())

#     # 解二次方程，满足：
#     # (u+v)=DeltaS, 且离差平方和变化=DeltaQ
#     # 令 v=DeltaS-u，可得 A u^2 + B u + C = 0
#     A = 2.0
#     B = (-2 * DeltaS + 2 * (xi - T_MEAN) - 2 * (xj - T_MEAN))
#     C = (DeltaS ** 2 + 2 * (xj - T_MEAN) * DeltaS) - DeltaQ
#     disc = B * B - 4 * A * C
#     if disc < 0:
#         raise RuntimeError("微调失败：判别式 < 0")

#     sqrt_disc = math.sqrt(disc)
#     u1 = (-B + sqrt_disc) / (2 * A)
#     u2 = (-B - sqrt_disc) / (2 * A)

#     def ok(u):
#         v = DeltaS - u
#         xi_new = xi + u
#         xj_new = xj + v
#         return (T_MIN <= xi_new <= T_MAX) and (T_MIN <= xj_new <= T_MAX)

#     # 选一个在范围内且幅度更小的解
#     candidates = []
#     for u in (u1, u2):
#         if ok(u):
#             candidates.append(u)
#     if not candidates:
#         raise RuntimeError("微调失败：两组解都会把值推到 min/max 之外")
#     u = min(candidates, key=lambda t: abs(t))
#     v = DeltaS - u

#     x_flat[hi_idx] = x_flat[hi_idx] + torch.tensor(u, dtype=DTYPE)
#     x_flat[lo_idx] = x_flat[lo_idx] + torch.tensor(v, dtype=DTYPE)
#     x = x_flat.view(SHAPE)

#     # 5) 最终校验并保存
#     print("min/max:", x.min().item(), x.max().item())
#     print("mean/std:", x.mean().item(), x.std(unbiased=True).item())
#     print("has_nan:", torch.isnan(x).any().item())
#     print("has_inf:", torch.isinf(x).any().item())

#     torch.save({"steering_activations": x.cpu()}, DST_PT)
#     print("saved:", DST_PT)


# if __name__ == "__main__":
#     main()
import os
import torch

# =========================
# 你要的“写死的参数”
# =========================
SRC_PT = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations_new_10.pt"
DST_PT = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations_random.pt"
SEED = 2025

# 如果 SRC_PT 能读到，就会优先用 SRC_PT 的 shape/dtype；否则用下面写死的默认值
SHAPE = (64, 256)
DTYPE = torch.float32
# =========================


def safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    shape = SHAPE
    dtype = DTYPE

    # 可选：从 SRC_PT 读取 shape/dtype（与你原脚本逻辑一致）
    try:
        obj = safe_load(SRC_PT)
        if isinstance(obj, dict) and "steering_activations" in obj:
            x0 = obj["steering_activations"]
            shape = tuple(x0.shape)
            dtype = x0.dtype
        else:
            print("[WARN] SRC_PT 格式不是 {'steering_activations': tensor}，将使用写死的 SHAPE/DTYPE 生成")
    except Exception as e:
        print(f"[INFO] 未读取 SRC_PT（{e}），直接按 SHAPE/DTYPE 生成")

    # 1) 生成近似 N(0,1) 的随机激活（样本均值/方差不强制精确对齐）
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)

    x = torch.randn(shape, generator=g, dtype=dtype, device="cpu")  # ~ N(0,1)

    # 2) 打印统计量检查
    print("shape/dtype:", tuple(x.shape), x.dtype)
    print("min/max:", x.min().item(), x.max().item())
    print("mean/std:", x.mean().item(), x.std(unbiased=True).item())
    print("has_nan:", torch.isnan(x).any().item())
    print("has_inf:", torch.isinf(x).any().item())

    # 3) 保存：格式与示例一致
    os.makedirs(os.path.dirname(DST_PT) or ".", exist_ok=True)
    torch.save({"steering_activations": x}, DST_PT)
    print("saved:", DST_PT)


if __name__ == "__main__":
    main()
