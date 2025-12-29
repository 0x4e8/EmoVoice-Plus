# 添加代码
import os, random, torch
import torchaudio
from collections import defaultdict

# 1) load CosyVoice/CosyVoice2 (你已经有 CosyVoice/CosyVoice2 类)
from examples.tts.utils.cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice

def find_first_modulelist(estimator):
    # 尽量自动找到 blocks
    for name in ["blocks", "layers", "transformer_blocks", "dit_blocks"]:
        if hasattr(estimator, name):
            m = getattr(estimator, name)
            if isinstance(m, torch.nn.ModuleList):
                return m
    for n, m in estimator.named_children():
        if isinstance(m, torch.nn.ModuleList):
            return m
    raise RuntimeError("Cannot find estimator blocks ModuleList; print(estimator) to locate it.")

def infer_step(args, kwargs, steps=32):
    # 你需要根据 estimator.forward 的签名微调这里
    # 常见关键字：t / time / sigma
    t = None
    for k in ["t", "time", "sigma"]:
        if k in kwargs and torch.is_tensor(kwargs[k]):
            t = kwargs[k]
            break
    if t is None:
        # fallback: 在 args 里找一个标量tensor
        for a in args:
            if torch.is_tensor(a) and a.numel() == 1:
                t = a
                break
    if t is None:
        return 0
    tt = float(t.flatten()[0].item())
    s = int(tt * steps)
    return max(0, min(steps - 1, s))

@torch.no_grad()
def collect_mean_acts(codec, texts, prompt_wavs, layer_indices, steps=32, sr=16000):
    """
    返回: acts[g, s] -> list of vectors(C)
    """
    estimator = codec.model.flow.decoder.estimator
    blocks = find_first_modulelist(estimator)

    # buckets[g][s] = list[tensor(C)]
    buckets = [[[] for _ in range(steps)] for _ in range(len(layer_indices))]

    handles = []
    for gi, li in enumerate(layer_indices):
        def make_hook(gidx):
            def hook(module, args, kwargs):
                # 取 residual stream（block input）
                x = args[0] if len(args) > 0 else kwargs.get("x", None)
                if x is None or (not torch.is_tensor(x)):
                    return
                step = infer_step(args, kwargs, steps=steps)

                # x: [B, T, C] 或 [B, C, T] 等，按实际调整
                if x.dim() == 3 and x.shape[-1] < x.shape[1]:
                    # [B, T, C]
                    v = x.mean(dim=1).mean(dim=0)  # -> [C]
                elif x.dim() == 3:
                    # [B, C, T]
                    v = x.mean(dim=2).mean(dim=0)  # -> [C]
                else:
                    return
                buckets[gidx][step].append(v.detach().float().cpu())
            return hook

        h = blocks[li].register_forward_pre_hook(make_hook(gi), with_kwargs=True)
        handles.append(h)

    # 触发一次完整 TTS（让 flow 跑起来）
    # 这里你可以用 codec.frontend.frontend_zero_shot 组 input，然后 codec.model.tts(...) 生成
    for text in texts:
        for wav in prompt_wavs:
            prompt_speech, _ = torchaudio.load(wav)  # -> [ch, T]
            prompt_speech = prompt_speech[:, :sr*10] # 可截断，省时间
            # 这里用你已有 frontend：
            model_in = codec.frontend.frontend_zero_shot(
                tts_text=text,
                prompt_text="",
                prompt_speech_16k=prompt_speech,
                resample_rate=codec.sample_rate,
            )
            # 只要跑到 flow，就会触发 hooks
            for _ in codec.model.tts(**model_in, stream=False, speed=1.0):
                pass

    for h in handles:
        h.remove()

    # mean
    out = torch.zeros(len(layer_indices), steps, 1)  # 先占位，后面按真实C改
    vecs = []
    for g in range(len(layer_indices)):
        row = []
        for s in range(steps):
            if len(buckets[g][s]) == 0:
                row.append(None)
            else:
                row.append(torch.stack(buckets[g][s], dim=0).mean(dim=0))
        vecs.append(row)
    return vecs  # list[G][S] of Tensor(C) or None

def stack_and_norm(delta_list):
    # delta_list: list[G][S] Tensor(C)
    G, S = len(delta_list), len(delta_list[0])
    C = delta_list[0][0].numel()
    out = torch.zeros(G, S, C)
    for g in range(G):
        for s in range(S):
            v = delta_list[g][s]
            v = v / (v.norm(p=2) + 1e-8)
            out[g, s] = v
    return out

def main():
    # ====== 你需要填：模型路径 & 数据列表 ======
    codec = CosyVoice2("/path/to/cosyvoice2_ckpt", load_jit=False, load_trt=False, fp16=False)

    layer_indices = [1, 6, 11, 16, 21]   # 先按论文“每隔5层”思路，具体看你estimator有多少层
    steps = 32

    texts = ["A short neutral sentence.", "Another sentence for synthesis."]  # 你可以放几十条随机英文句子
    neutral_wavs = ['/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/audio/neutral/gpt4o_6000_neutral_verse.wav']  # 一批 neutral prompt speech wav 路径
    emo_wavs = {
        "happy": ['/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/audio/happy/gpt4o_100_happy_verse.wav'],
        "sad":   ['/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/audio/happy/gpt4o_100_happy_verse.wav'],
        "angry": ['/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/audio/happy/gpt4o_100_happy_verse.wav'],
    }

    # 先算 neutral 的均值激活
    A_neu = collect_mean_acts(codec, texts, neutral_wavs, layer_indices, steps=steps)

    out = {"meta": {"steps": steps, "layers": layer_indices}}
    for emo, wavs in emo_wavs.items():
        A_emo = collect_mean_acts(codec, texts, wavs, layer_indices, steps=steps)

        # delta = mean(A_emo) - mean(A_neu)
        delta = []
        for g in range(len(layer_indices)):
            delta.append([A_emo[g][s] - A_neu[g][s] for s in range(steps)])

        out[emo] = stack_and_norm(delta)

    torch.save(out, "steering_activations.pt")

if __name__ == "__main__":
    main()
