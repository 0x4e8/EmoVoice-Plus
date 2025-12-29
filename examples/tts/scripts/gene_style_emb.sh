torchrun --nproc_per_node 2 --master_port 29509 \
  examples/tts/utils/precompute_style_emb.py \
  --jsonl /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train_and_val.jsonl \
  --out_pt /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train_and_val.style_emb.pt \
  --audio_key target_wav \
  --audio_root /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice \
  --model /data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt/emotion2vec_plus_large \
  --chunk_size 512 --batch_size 64 --cleanup
