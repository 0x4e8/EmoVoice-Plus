python build_neutral_happy_sad_triplets.py \
  --api_base "https://api.openai-proxy.org/v1" \
  --api_key  "sk-*****" \
  --out_dir ./triplets_out \
  --num_samples 15 \
  --sdk_max_retries 0 \
  --timeout_read 60 \
  --max_task_retries 3 \
  --llm_model gpt-4o \
  --tts_model gpt-4o-mini-tts \
  --asr_model gpt-4o-mini-transcribe  \
  --voices "ballad,verse,nova"

