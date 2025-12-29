# python - <<'PY'
# import random
# from pathlib import Path

# src = Path("/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/test.jsonl")
# dst = Path("/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/test_mini.jsonl")
# k = 100

# random.seed(42)  # 想每次不一样就删掉这行

# reservoir = []
# with src.open("r", encoding="utf-8") as f:
#     for i, line in enumerate(f, start=1):
#         if i <= k:
#             reservoir.append(line)
#         else:
#             j = random.randint(1, i)
#             if j <= k:
#                 reservoir[j - 1] = line

# if not reservoir:
#     raise RuntimeError(f"Empty file: {src}")

# random.shuffle(reservoir)  # 打乱写出顺序（可选）
# dst.parent.mkdir(parents=True, exist_ok=True)
# with dst.open("w", encoding="utf-8") as g:
#     g.writelines(reservoir)

# print(f"Wrote {len(reservoir)} lines to: {dst}")
# PY
export PYTHONPATH=$PYTHONPATH:/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/src
# python examples/tts/build_steering_activations.py \
#   --codec_decoder_path /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt/ckpts/CosyVoice/CosyVoice-300M-SFT \
#   --cosyvoice_version 1 \
#   --neutral_list /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/data/neutral.txt \
#   --emotion_json /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/data/emotions.json \
#   --base_token_wav /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/audio/neutral/gpt4o_6000_neutral_verse.wav \
#   --base_token_len 200 \
#   --steps 32 \
#   --block_group 5 \
#   --out /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/steering_activations.pt


# 生成 .pt
python examples/tts/build_steering_activations.py \
  --codec_decoder_path /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt/ckpts/CosyVoice/CosyVoice-300M-SFT \
  --cosyvoice_version 1 \
  --prompt_lists_json /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/data/audio_paths.json \
  --out_path /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/steering_activations_new_10.pt \
  --token_max_len 512 \
  --max_refs_per_emotion 10 \
  --device cuda:2


python - <<'PY'
import torch
p="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/steering_activations_new_10.pt"
x=torch.load(p, map_location="cpu")
print(type(x))
if isinstance(x, dict): print("keys:", list(x.keys()))
else: print("shape:", getattr(x, "shape", None))
PY

python - <<'PY'
import torch
x=torch.load("/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/steering_activations_new_10.pt", map_location="cpu")
print("steer_dir emotions:", list(x["steer_dir"].keys())[:50])
print("steer_vec emotions:", list(x["steer_vec"].keys())[:50])
PY


# 转换.pt
python - <<'PY'
import torch
src="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/steering_activations_new_10.pt"
out="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/happy_steering_activations_new_10.pt"

x=torch.load(src, map_location="cpu")
t=x["steer_dir"]["happy"]          # 或 x["steer_vec"]["happy"]
torch.save({"steering_activations": t.cpu()}, out)
print("saved:", out, "shape:", tuple(t.shape))
PY

