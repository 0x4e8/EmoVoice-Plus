
# ==============================================================================
# 有用
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# build_neutral_happy_sad_triplets.py

# 生成 neutral-happy-sad 三元对（同一文本三种情绪三条 wav），并用 ASR + WER 过滤。

# 修复点：
# - Step1 词数不达标/JSON 解析失败：允许重试，不再直接 FATAL
# - audio.speech.create 参数：使用 response_format="wav"（官方），并兼容旧代理
# - 保留：timeout、日志、429/5xx 退避，避免“卡住无输出”
# """

# import argparse
# import base64
# import json
# import os
# import random
# import re
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# import httpx
# import openai
# from openai import OpenAI


# # ----------------------------
# # WER (no external deps)
# # ----------------------------
# _WORD_RE = re.compile(r"[A-Za-z']+")

# def _tokenize_for_wer(s: str) -> List[str]:
#     s = s.lower()
#     return _WORD_RE.findall(s)

# def wer(ref: str, hyp: str) -> float:
#     r = _tokenize_for_wer(ref)
#     h = _tokenize_for_wer(hyp)
#     if not r and not h:
#         return 0.0
#     if not r:
#         return 1.0

#     dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
#     for i in range(len(r) + 1):
#         dp[i][0] = i
#     for j in range(len(h) + 1):
#         dp[0][j] = j
#     for i in range(1, len(r) + 1):
#         for j in range(1, len(h) + 1):
#             cost = 0 if r[i - 1] == h[j - 1] else 1
#             dp[i][j] = min(
#                 dp[i - 1][j] + 1,
#                 dp[i][j - 1] + 1,
#                 dp[i - 1][j - 1] + cost
#             )
#     return dp[len(r)][len(h)] / max(1, len(r))

# def count_words(s: str) -> int:
#     return len(_WORD_RE.findall(s))


# # ----------------------------
# # Utils
# # ----------------------------
# def log(msg: str) -> None:
#     print(msg, flush=True)

# def normalize_api_base(api_base: str) -> str:
#     api_base = (api_base or "").strip().rstrip("/")
#     if not api_base:
#         return api_base
#     if not api_base.endswith("/v1"):
#         api_base += "/v1"
#     return api_base

# def save_b64_audio_to_file(b64_data: str, out_path: Path) -> None:
#     audio_bytes = base64.b64decode(b64_data)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_bytes(audio_bytes)

# def safe_unlink(p: Path) -> None:
#     try:
#         p.unlink(missing_ok=True)
#     except Exception:
#         pass

# def is_rate_limit_error(e: Exception) -> bool:
#     if isinstance(e, getattr(openai, "RateLimitError", ())):
#         return True
#     status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
#     if status == 429:
#         return True
#     return "429" in repr(e) or "Rate limit" in repr(e) or "Too Many Requests" in repr(e)

# def is_transient_error(e: Exception) -> bool:
#     if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
#         return True
#     if isinstance(e, getattr(openai, "APIConnectionError", ())):
#         return True
#     status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
#     return status is not None and 500 <= int(status) <= 599

# def sleep_backoff(base: float, attempt: int, jitter: float = 0.2, cap: float = 10.0) -> None:
#     t = min(cap, base * (2 ** (attempt - 1)))
#     t = t + random.uniform(0, jitter)
#     time.sleep(t)


# # ----------------------------
# # Generation config
# # ----------------------------
# DEFAULT_VOICES = ["ash", "coral", "onyx", "shimmer", "sage"]
# TEXT_STYLES = ["prose", "dialogue", "observation"]


# # ----------------------------
# # Rich emotion prosody hints (used in TTS prompts)
# # ----------------------------
# EMOTION_RICH_HINT = {
#     "neutral": (
#         "Neutral delivery: steady pace, moderate pitch, minimal expressiveness, clean articulation."
#     ),
#     "happy": (
#         "Rich happy delivery: brighter timbre, higher pitch range, wider intonation, slightly faster pace, "
#         "higher energy, smiling voice, lively rhythm. Use prosody-only emphasis on a few positive-sounding words. "
#         "Do NOT laugh, do NOT add fillers, and do NOT change any words."
#     ),
#     "sad": (
#         "Rich sad delivery: softer volume, lower pitch range, narrower intonation, slightly slower pace, "
#         "lower energy, gentle breathiness, subtle trembling quality, and softer/longer sentence endings. "
#         "Use small natural pauses between clauses. Do NOT sob/cry, do NOT add fillers, and do NOT change any words."
#     ),
# }

# def get_emotion_hint(emotion: str) -> str:
#     return EMOTION_RICH_HINT.get((emotion or "").lower(), "")


# # ----------------------------
# # LLM helpers
# # ----------------------------
# def extract_json_loose(s: str) -> Dict:
#     try:
#         return json.loads(s)
#     except Exception:
#         pass
#     m = re.search(r"\{.*\}", s, flags=re.S)
#     if not m:
#         raise ValueError(f"No JSON object found. Raw={s[:200]}...")
#     return json.loads(m.group(0))

# def gen_text_and_descriptions(client: OpenAI, llm_model: str, style: str, seed: int) -> Tuple[str, str, str, str]:
#     sys = (
#         "You are generating training data for emotional speech synthesis.\n"
#         "Return ONLY valid JSON. No markdown, no extra keys.\n"
#         "Constraints:\n"
#         "- text: 18–22 words (MUST be within 15–25), English, rich sensory detail, not clichéd.\n"
#         "- Before outputting JSON, silently COUNT the words and rewrite until the word count is within 15–25.\n"
#         "- style must match the requested category.\n"
#         "- Up to TWO words in ALL CAPS allowed to indicate stress.\n"
#         "- descriptions: ONE sentence; present participle verb phrase (start with 'Conveying/Expressing/Emanating/Evoking' etc);\n"
#         "  focus ONLY on vocal affect; do NOT mention events, people, places, or specific context.\n"
#         "- For desc_happy and desc_sad: include at least THREE prosody cues among pitch/pace/energy/intonation/pauses/brightness/breathiness.\n"
#     )
#     user = (
#         f"Style category: {style}\n"
#         "Generate ONE emotionally-ambiguous sentence that can be spoken as neutral, happy, and sad without changing the words.\n"
#         "Then generate three vocal-affect descriptions for neutral/happy/sad.\n"
#         "Output JSON with keys exactly:\n"
#         "{\"text\":...,\"desc_neutral\":...,\"desc_happy\":...,\"desc_sad\":...}\n"
#         f"Seed hint (may ignore): {seed}"
#     )

#     resp = client.chat.completions.create(
#         model=llm_model,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0.7,
#         max_tokens=450,
#     )
#     content = resp.choices[0].message.content or ""
#     obj = extract_json_loose(content)

#     text = str(obj["text"]).strip()
#     dn = str(obj["desc_neutral"]).strip()
#     dh = str(obj["desc_happy"]).strip()
#     ds = str(obj["desc_sad"]).strip()

#     wc = count_words(text)
#     if not (15 <= wc <= 25):
#         raise ValueError(f"text word count={wc} not in [15,25]. text={text}")
#     return text, dn, dh, ds

# def paraphrase_description(client: OpenAI, llm_model: str, desc: str, n: int) -> List[str]:
#     sys = (
#         "Rewrite emotion descriptions.\n"
#         "Each output must be ONE sentence, a present participle verb phrase, focusing only on vocal affect.\n"
#         "Avoid mentioning events/places/people/context.\n"
#         "Return ONLY JSON: {\"alts\":[...]} with exactly n items."
#     )
#     user = f"Original description: {desc}\nGenerate {n} rephrasings."
#     resp = client.chat.completions.create(
#         model=llm_model,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0.8,
#         max_tokens=220,
#     )
#     obj = extract_json_loose(resp.choices[0].message.content or "")
#     alts = [str(x).strip() for x in obj.get("alts", []) if str(x).strip()]
#     alts = [a for a in alts if a.lower() != desc.lower()]
#     return alts[:n]


# # ----------------------------
# # Audio generation
# # ----------------------------
# def is_speech_tts_model(model_name: str) -> bool:
#     return model_name.endswith("-tts") or model_name in {"tts-1", "tts-1-hd"}

# def synthesize_with_chat_audio(
#     client: OpenAI,
#     tts_model: str,
#     voice: str,
#     text: str,
#     desc: str,
#     wav_path: Path,
#     emotion: str,
# ) -> None:
#     hint = get_emotion_hint(emotion)
#     prompt = (
#         f"Emotion description: {desc}\n"
#         f"Prosody guidance: {hint}\n"
#         f"Read the sentence verbatim (no extra words): {text}"
#     )
#     resp = client.chat.completions.create(
#         model=tts_model,
#         modalities=["text", "audio"],
#         audio={"voice": voice, "format": "wav"},
#         messages=[
#             {"role": "system", "content": "Generate audio only. Speak the sentence verbatim. Never speak instructions or meta text."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.3,
#     )
#     audio = getattr(resp.choices[0].message, "audio", None)
#     if not audio or not getattr(audio, "data", None):
#         raise RuntimeError("No audio data returned (proxy/model may not support chat-audio modalities).")
#     save_b64_audio_to_file(audio.data, wav_path)

# def synthesize_with_speech_endpoint(
#     client: OpenAI,
#     tts_model: str,
#     voice: str,
#     text: str,
#     desc: str,
#     wav_path: Path,
#     emotion: str,
# ) -> None:
#     hint = get_emotion_hint(emotion)
#     instructions = (
#         f"Emotion description: {desc}\n"
#         f"{hint}\n"
#         "Speak naturally in English. Read the text exactly; do not add or omit words."
#     )

#     wav_path.parent.mkdir(parents=True, exist_ok=True)

#     # ✅ 兼容不同代理/SDK：优先使用官方 response_format；若不支持再降级
#     try:
#         audio = client.audio.speech.create(
#             model=tts_model,
#             voice=voice,
#             input=text,
#             response_format="wav",
#             instructions=instructions,
#         )
#     except TypeError as e:
#         msg = str(e)

#         # 1) 某些实现可能不支持 instructions：去掉 instructions 再试
#         if "instructions" in msg:
#             try:
#                 audio = client.audio.speech.create(
#                     model=tts_model,
#                     voice=voice,
#                     input=text,
#                     response_format="wav",
#                 )
#             except TypeError:
#                 # 2) 兜底：某些旧实现用 format 字段
#                 audio = client.audio.speech.create(
#                     model=tts_model,
#                     voice=voice,
#                     input=text,
#                     format="wav",
#                 )
#         else:
#             # 2) 兜底：某些旧实现用 format 字段
#             audio = client.audio.speech.create(
#                 model=tts_model,
#                 voice=voice,
#                 input=text,
#                 format="wav",
#                 instructions=instructions,
#             )

#     # 兼容不同返回类型
#     if hasattr(audio, "write_to_file"):
#         audio.write_to_file(str(wav_path))
#     elif hasattr(audio, "stream_to_file"):
#         audio.stream_to_file(str(wav_path))
#     elif isinstance(audio, (bytes, bytearray)):
#         wav_path.write_bytes(bytes(audio))
#     else:
#         wav_path.write_bytes(bytes(audio))

# def transcribe(client: OpenAI, asr_model: str, wav_path: Path) -> str:
#     with wav_path.open("rb") as f:
#         out = client.audio.transcriptions.create(
#             model=asr_model,
#             file=f,
#             response_format="text",
#         )
#     if isinstance(out, str):
#         return out.strip()
#     return (getattr(out, "text", None) or str(out)).strip()


# # ----------------------------
# # Request wrapper with controlled retries
# # ----------------------------
# def call_with_controlled_retries(
#     fn,
#     *,
#     what: str,
#     max_tries: int,
#     base_backoff: float,
#     retry_exceptions: Tuple[type, ...] = (),
# ):
#     """
#     retry_exceptions: 允许把“业务校验失败”（如词数不达标、JSON 解析失败）当作可重试错误。
#     """
#     last_err: Optional[Exception] = None
#     for attempt in range(1, max_tries + 1):
#         try:
#             return fn()
#         except Exception as e:
#             last_err = e
#             log(f"ERROR during {what} attempt={attempt}/{max_tries}: {repr(e)}")

#             # ✅ 业务层错误也可重试（短退避）
#             if retry_exceptions and isinstance(e, retry_exceptions):
#                 sleep_backoff(max(0.2, base_backoff), attempt, cap=2.0)
#                 continue

#             if is_rate_limit_error(e):
#                 sleep_backoff(base_backoff, attempt, cap=12.0)
#                 continue
#             if is_transient_error(e):
#                 sleep_backoff(base_backoff, attempt, cap=6.0)
#                 continue
#             raise
#     raise RuntimeError(f"Failed {what} after {max_tries} tries. Last error: {repr(last_err)}")


# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out_dir", type=str, required=True)
#     ap.add_argument("--num_samples", type=int, default=100)

#     ap.add_argument("--llm_model", type=str, default="gpt-4o")
#     ap.add_argument("--tts_model", type=str, default="gpt-4o-mini-tts")
#     ap.add_argument("--asr_model", type=str, default="gpt-4o-mini-transcribe")

#     ap.add_argument("--voices", type=str, default=",".join(DEFAULT_VOICES))
#     ap.add_argument("--wer_threshold", type=float, default=0.08)

#     ap.add_argument("--max_task_retries", type=int, default=3, help="Per-step retries (LLM/TTS/ASR).")
#     ap.add_argument("--backoff_s", type=float, default=0.6, help="Base backoff seconds for 429/5xx.")

#     ap.add_argument("--api_base", type=str, default=os.getenv("CLOSEAI_API_BASE", ""))
#     ap.add_argument("--api_key", type=str, default=os.getenv("CLOSEAI_API_KEY", ""))

#     ap.add_argument("--sdk_max_retries", type=int, default=0)
#     ap.add_argument("--timeout_connect", type=float, default=10.0)
#     ap.add_argument("--timeout_read", type=float, default=60.0)
#     ap.add_argument("--timeout_write", type=float, default=60.0)
#     ap.add_argument("--timeout_pool", type=float, default=10.0)

#     ap.add_argument("--augment_desc", action="store_true", help="Enable paraphrase augmentation (adds many requests).")
#     ap.add_argument("--desc_alt_n", type=int, default=2)

#     ap.add_argument("--keep_failed_wav", action="store_true", help="Keep wav files even if WER fails.")
#     ap.add_argument("--fail_fast", action="store_true", help="Stop immediately on an unrecoverable error.")

#     args = ap.parse_args()

#     api_base = normalize_api_base(args.api_base)
#     api_key = args.api_key
#     if not api_base or not api_key:
#         raise ValueError("api_base/api_key 不能为空：请传 --api_base/--api_key 或设置环境变量 CLOSEAI_API_BASE/CLOSEAI_API_KEY")

#     timeout = httpx.Timeout(
#         connect=args.timeout_connect,
#         read=args.timeout_read,
#         write=args.timeout_write,
#         pool=args.timeout_pool,
#     )
#     http_client = httpx.Client(timeout=timeout)

#     client = OpenAI(
#         base_url=api_base,
#         api_key=api_key,
#         timeout=timeout,
#         max_retries=args.sdk_max_retries,
#         http_client=http_client,
#     )

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     manifest_path = out_dir / "manifest.jsonl"

#     voices = [v.strip() for v in args.voices.split(",") if v.strip()]
#     if not voices:
#         raise ValueError("voices 为空。")

#     done = 0
#     if manifest_path.exists():
#         with manifest_path.open("r", encoding="utf-8") as f:
#             done = sum(1 for _ in f)

#     log(f"api_base={api_base}")
#     log(f"resume from {done}, target={args.num_samples}")
#     log(f"tts_mode={'speech_endpoint' if is_speech_tts_model(args.tts_model) else 'chat_audio'}")
#     log(f"sdk_max_retries={args.sdk_max_retries}, timeout={timeout}")

#     for idx in range(done, args.num_samples):
#         uid = f"{idx:06d}"
#         style = TEXT_STYLES[idx % len(TEXT_STYLES)]
#         voice = voices[idx % len(voices)]

#         try:
#             log(f"[{uid}] step1: gen_text_and_desc (style={style}, voice={voice})")
#             text, dn, dh, ds = call_with_controlled_retries(
#                 lambda: gen_text_and_descriptions(client, args.llm_model, style, seed=idx),
#                 what=f"{uid}/gen_text_and_desc",
#                 max_tries=args.max_task_retries,
#                 base_backoff=args.backoff_s,
#                 retry_exceptions=(ValueError, KeyError, json.JSONDecodeError),
#             )

#             alt_dn, alt_dh, alt_ds = [], [], []
#             if args.augment_desc:
#                 log(f"[{uid}] step1b: paraphrase_desc x3")
#                 alt_dn = call_with_controlled_retries(
#                     lambda: paraphrase_description(client, args.llm_model, dn, n=args.desc_alt_n),
#                     what=f"{uid}/paraphrase_neutral",
#                     max_tries=args.max_task_retries,
#                     base_backoff=args.backoff_s,
#                     retry_exceptions=(ValueError, KeyError, json.JSONDecodeError),
#                 )
#                 alt_dh = call_with_controlled_retries(
#                     lambda: paraphrase_description(client, args.llm_model, dh, n=args.desc_alt_n),
#                     what=f"{uid}/paraphrase_happy",
#                     max_tries=args.max_task_retries,
#                     base_backoff=args.backoff_s,
#                     retry_exceptions=(ValueError, KeyError, json.JSONDecodeError),
#                 )
#                 alt_ds = call_with_controlled_retries(
#                     lambda: paraphrase_description(client, args.llm_model, ds, n=args.desc_alt_n),
#                     what=f"{uid}/paraphrase_sad",
#                     max_tries=args.max_task_retries,
#                     base_backoff=args.backoff_s,
#                     retry_exceptions=(ValueError, KeyError, json.JSONDecodeError),
#                 )

#             wav_neu = out_dir / "wav" / "neutral" / f"{uid}_{voice}.wav"
#             wav_hap = out_dir / "wav" / "happy" / f"{uid}_{voice}.wav"
#             wav_sad = out_dir / "wav" / "sad" / f"{uid}_{voice}.wav"

#             def synth_and_asr(emotion: str, desc: str, wav_path: Path) -> Tuple[str, float]:
#                 log(f"[{uid}] step2: synth+asr {emotion}")

#                 def _do():
#                     if is_speech_tts_model(args.tts_model):
#                         synthesize_with_speech_endpoint(
#                             client, args.tts_model, voice, text, desc, wav_path, emotion=emotion
#                         )
#                     else:
#                         synthesize_with_chat_audio(
#                             client, args.tts_model, voice, text, desc, wav_path, emotion=emotion
#                         )

#                     hyp = transcribe(client, args.asr_model, wav_path)
#                     score = wer(text, hyp)
#                     return hyp, score

#                 hyp, score = call_with_controlled_retries(
#                     _do,
#                     what=f"{uid}/synth_asr_{emotion}",
#                     max_tries=args.max_task_retries,
#                     base_backoff=args.backoff_s,
#                 )
#                 return hyp, score

#             asr_neu, wer_neu = synth_and_asr("neutral", dn, wav_neu)
#             asr_hap, wer_hap = synth_and_asr("happy", dh, wav_hap)
#             asr_sad, wer_sad = synth_and_asr("sad", ds, wav_sad)

#             kept = (wer_neu <= args.wer_threshold and wer_hap <= args.wer_threshold and wer_sad <= args.wer_threshold)

#             if not kept and not args.keep_failed_wav:
#                 safe_unlink(wav_neu)
#                 safe_unlink(wav_hap)
#                 safe_unlink(wav_sad)

#             item = {
#                 "id": uid,
#                 "style": style,
#                 "voice": voice,
#                 "text": text,
#                 "desc_neutral": dn,
#                 "desc_happy": dh,
#                 "desc_sad": ds,
#                 "desc_neutral_alts": alt_dn,
#                 "desc_happy_alts": alt_dh,
#                 "desc_sad_alts": alt_ds,
#                 "wav_neutral": str(wav_neu.relative_to(out_dir)) if kept or args.keep_failed_wav else None,
#                 "wav_happy": str(wav_hap.relative_to(out_dir)) if kept or args.keep_failed_wav else None,
#                 "wav_sad": str(wav_sad.relative_to(out_dir)) if kept or args.keep_failed_wav else None,
#                 "asr_neutral": asr_neu,
#                 "asr_happy": asr_hap,
#                 "asr_sad": asr_sad,
#                 "wer_neutral": wer_neu,
#                 "wer_happy": wer_hap,
#                 "wer_sad": wer_sad,
#                 "kept": kept,
#                 "api_base": api_base,
#                 "models": {"llm": args.llm_model, "tts": args.tts_model, "asr": args.asr_model},
#             }

#             with manifest_path.open("a", encoding="utf-8") as f:
#                 f.write(json.dumps(item, ensure_ascii=False) + "\n")

#             log(f"[{uid}] kept={kept} | WER n/h/s = {wer_neu:.3f}/{wer_hap:.3f}/{wer_sad:.3f}")

#         except Exception as e:
#             log(f"[{uid}] FATAL ERROR: {repr(e)}")
#             if args.fail_fast:
#                 raise
#             continue


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_neutral_happy_sad_triplets.py

生成 neutral-happy-sad 三元对（同一文本三种情绪三条 wav），并用 ASR + 对齐统计过滤。

本版按你的 3 点需求修改：
1) Step1 通过更强 Prompt，让 gpt-4o 生成“情绪可塑性”更强的文本（避免指责/时间压力/强语用冲突等）。
2) 情绪控制从“类别”升级为“强度 + 唤醒度”：每个情绪输出 {desc, arousal, intensity}，TTS 指令里显式注入。
3) 允许更自然连读/轻气声：happy/sad 的 WER 阈值更宽，但严格要求“不丢词”（del=0），并在不达标时自动重试合成。

兼容点：
- Step1 词数/JSON 失败：允许重试，不直接 FATAL
- audio.speech.create：优先使用 response_format="wav"，并对部分代理做降级兜底
- 保留 timeout、日志、429/5xx 退避，避免“卡住无输出”
"""

import argparse
import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import httpx
import openai
from openai import OpenAI


# ----------------------------
# Tokenization / Alignment stats
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z']+")

def _tokenize_for_wer(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def count_words(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))

def wer_stats(ref: str, hyp: str) -> Tuple[float, int, int, int, int]:
    """
    Returns: (wer, ins, del, sub, ref_len)
    Using standard DP Levenshtein alignment counts.
    """
    r = _tokenize_for_wer(ref)
    h = _tokenize_for_wer(hyp)
    if not r and not h:
        return 0.0, 0, 0, 0, 0
    if not r:
        return 1.0, len(h), 0, 0, 0

    # dp cost
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    # backtrace ops: 0=match/sub, 1=del, 2=ins
    bt = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]

    for i in range(1, len(r) + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, len(h) + 1):
        dp[0][j] = j
        bt[0][j] = 2

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost_sub = 0 if r[i - 1] == h[j - 1] else 1
            cand = [
                (dp[i - 1][j] + 1, 1),            # del
                (dp[i][j - 1] + 1, 2),            # ins
                (dp[i - 1][j - 1] + cost_sub, 0), # sub/match
            ]
            best = min(cand, key=lambda x: x[0])
            dp[i][j], bt[i][j] = best

    # backtrace counts
    i, j = len(r), len(h)
    ins = dele = sub = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 1:
            dele += 1
            i -= 1
        elif op == 2:
            ins += 1
            j -= 1
        else:
            # match/sub
            if i > 0 and j > 0 and r[i - 1] != h[j - 1]:
                sub += 1
            i -= 1
            j -= 1

    wer = (ins + dele + sub) / max(1, len(r))
    return wer, ins, dele, sub, len(r)


# ----------------------------
# Utils
# ----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def normalize_api_base(api_base: str) -> str:
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return api_base
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    return api_base

def save_b64_audio_to_file(b64_data: str, out_path: Path) -> None:
    audio_bytes = base64.b64decode(b64_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)

def safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, getattr(openai, "RateLimitError", ())):
        return True
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status == 429:
        return True
    return "429" in repr(e) or "Rate limit" in repr(e) or "Too Many Requests" in repr(e)

def is_transient_error(e: Exception) -> bool:
    if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(e, getattr(openai, "APIConnectionError", ())):
        return True
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    return status is not None and 500 <= int(status) <= 599

def sleep_backoff(base: float, attempt: int, jitter: float = 0.2, cap: float = 10.0) -> None:
    t = min(cap, base * (2 ** (attempt - 1)))
    t = t + random.uniform(0, jitter)
    time.sleep(t)


# ----------------------------
# Generation config
# ----------------------------
# 官方支持的 voices（你脚本里用的 ballad/verse/nova 是可用的）
DEFAULT_VOICES = ["ash", "coral", "onyx", "shimmer", "sage"]
TEXT_STYLES = ["prose", "dialogue", "observation"]


# ----------------------------
# Emotion control (category -> arousal + intensity)
# ----------------------------
@dataclass
class EmotionControl:
    desc: str
    arousal: float   # 0~1
    intensity: float # 0~1

def _clamp01(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return max(0.0, min(1.0, v))

def prosody_guidance(emotion: str, arousal: float, intensity: float) -> str:
    """
    Convert (arousal, intensity) into concrete prosody hints.
    """
    emotion = (emotion or "").lower().strip()
    a = _clamp01(arousal, 0.5)
    i = _clamp01(intensity, 0.5)

    # pace / pitch range from arousal
    if a >= 0.7:
        pace = "slightly faster pace"
        pitch = "wide pitch range"
        rhythm = "lively rhythmic phrasing"
    elif a <= 0.35:
        pace = "slightly slower pace"
        pitch = "narrower pitch range"
        rhythm = "gentler, more even rhythm"
    else:
        pace = "steady pace"
        pitch = "moderate pitch range"
        rhythm = "natural, balanced rhythm"

    # energy / emphasis from intensity
    if i >= 0.75:
        energy = "high expressive energy"
        emphasis = "clear prosodic emphasis on a few key words (prosody-only)"
    elif i <= 0.35:
        energy = "subtle expressive energy"
        emphasis = "minimal emphasis, restrained expressiveness"
    else:
        energy = "moderate expressive energy"
        emphasis = "light prosodic emphasis where it sounds natural"

    if emotion == "neutral":
        return (
            "Neutral delivery: clean articulation, controlled intonation, minimal emotional coloring. "
            f"Target arousal={a:.2f}, intensity={i:.2f}. Keep {pace}, {pitch}, {rhythm}; {energy}; {emphasis}."
        )
    if emotion == "happy":
        return (
            "Happy delivery (positive valence): brighter timbre, lifted intonation, smiling voice quality. "
            f"Target arousal={a:.2f}, intensity={i:.2f}. Use {pace}, {pitch}, {rhythm}; {energy}; {emphasis}. "
            "Allow natural connected speech and light breathiness if it helps expressiveness. "
            "Do NOT laugh, do NOT add fillers, and do NOT change any words."
        )
    if emotion == "sad":
        return (
            "Sad delivery (negative valence): softer volume, darker timbre, gentle breathiness, softer endings. "
            f"Target arousal={a:.2f}, intensity={i:.2f}. Use {pace}, {pitch}, {rhythm}; {energy}; {emphasis}. "
            "Allow natural connected speech and light breathiness if it helps realism. "
            "Do NOT sob/cry, do NOT add fillers, and do NOT change any words."
        )
    return f"Target arousal={a:.2f}, intensity={i:.2f}. {pace}; {pitch}; {energy}."


# ----------------------------
# LLM helpers
# ----------------------------
def extract_json_loose(s: str) -> Dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object found. Raw={s[:200]}...")
    return json.loads(m.group(0))

# 简单 blacklist：避免强语用冲突/责备/时间压力导致“happy 带不动”
_BAD_PATTERNS = [
    r"\bby now\b",
    r"\byou (?:didn't|did not|forgot|never)\b",
    r"\bmaybe you\b",
    r"\byou should\b|\byou must\b|\byou have to\b",
    r"\bshould have\b|\bmust have\b",
    r"\balways\b|\bnever\b",
    r"\bsorry\b|\bapolog",
    r"\bcall(?:ed)?\b.*\bby now\b",
    r"\bhappy\b|\bsad\b|\bangry\b|\bcry\b|\blaugh\b",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS), flags=re.I)

def validate_plastic_text(text: str) -> None:
    wc = count_words(text)
    if not (12 <= wc <= 25):
        raise ValueError(f"text word count={wc} not in [12,25]. text={text}")
    if _BAD_RE.search(text or ""):
        raise ValueError(f"text not emotion-plastic (matched taboo pattern). text={text}")
    # 让韵律更有空间：建议至少有一个逗号/分号/破折号
    if not re.search(r"[,;—-]", text):
        raise ValueError(f"text lacks prosodic punctuation (no comma/semicolon/dash). text={text}")

def _parse_controls(obj: Dict) -> Tuple[str, EmotionControl, EmotionControl, EmotionControl]:
    """
    Accept either:
    A) legacy flat keys: desc_neutral/desc_happy/desc_sad
    B) new nested keys: neutral/happy/sad objects with desc/arousal/intensity
    """
    text = str(obj.get("text", "")).strip()
    if not text:
        raise KeyError("Missing 'text'.")

    if "neutral" in obj and "happy" in obj and "sad" in obj:
        def _one(key: str, da: float, di: float) -> EmotionControl:
            o = obj.get(key, {}) or {}
            desc = str(o.get("desc", "")).strip()
            if not desc:
                raise KeyError(f"Missing {key}.desc")
            ar = _clamp01(o.get("arousal", da), da)
            it = _clamp01(o.get("intensity", di), di)
            return EmotionControl(desc=desc, arousal=ar, intensity=it)

        neu = _one("neutral", 0.45, 0.20)
        hap = _one("happy",   0.80, 0.80)
        sad = _one("sad",     0.35, 0.80)
        return text, neu, hap, sad

    # fallback legacy
    dn = str(obj["desc_neutral"]).strip()
    dh = str(obj["desc_happy"]).strip()
    ds = str(obj["desc_sad"]).strip()
    neu = EmotionControl(desc=dn, arousal=0.45, intensity=0.20)
    hap = EmotionControl(desc=dh, arousal=0.80, intensity=0.80)
    sad = EmotionControl(desc=ds, arousal=0.35, intensity=0.80)
    return text, neu, hap, sad

def gen_text_and_controls(client: OpenAI, llm_model: str, style: str, seed: int) -> Tuple[str, EmotionControl, EmotionControl, EmotionControl]:
    """
    Step1: 生成“情绪可塑性”更强的文本 + 三情绪控制（desc + arousal + intensity）
    """
    sys = (
        "You are generating training data for emotional speech synthesis.\n"
        "Return ONLY valid JSON. No markdown.\n\n"
        "GOAL: produce an emotion-plastic sentence that can be performed as neutral, happy, or sad WITHOUT sounding sarcastic.\n\n"
        "TEXT CONSTRAINTS:\n"
        "- English, 18–22 words (MUST be within 15–25). Before outputting JSON, silently count words and rewrite until satisfied.\n"
        "- Must include at least one comma or semicolon (prosodic structure).\n"
        "- Avoid accusatory or obligation language that locks affect (avoid: 'by now', 'you should', 'you must', 'maybe you', 'you didn't', apologies).\n"
        "- Avoid explicit emotion words (happy/sad/angry/cry/laugh) and avoid overt conflict or blame.\n"
        "- Keep semantics mildly ambiguous and imagery-friendly; allow performance to carry emotion.\n"
        "- No slang, no profanity.\n\n"
        "STYLE:\n"
        "- prose: descriptive but not poetic clichés.\n"
        "- observation: sensory observation with gentle ambiguity.\n"
        "- dialogue: a single line that is NOT a complaint and NOT time-pressure; should sound plausible in neutral/happy/sad.\n\n"
        "EMOTION CONTROL OUTPUT:\n"
        "- Provide for each emotion: desc, arousal, intensity.\n"
        "- desc: ONE sentence; present participle verb phrase (start with 'Conveying/Expressing/Emanating/Evoking' etc);\n"
        "  focus ONLY on vocal affect/prosody (pitch/pace/energy/intonation/pauses/brightness/breathiness), NO events/people/places/context.\n"
        "- arousal: number in [0,1]. intensity: number in [0,1].\n"
        "  neutral: arousal 0.35–0.60, intensity 0.10–0.35\n"
        "  happy:   arousal 0.65–0.95, intensity 0.55–0.95\n"
        "  sad:     arousal 0.20–0.55, intensity 0.55–0.95\n"
    )
    user = (
        f"Style category: {style}\n"
        "Output JSON with keys exactly:\n"
        "{"
        "\"text\": \"...\", "
        "\"neutral\": {\"desc\":\"...\",\"arousal\":0.xx,\"intensity\":0.xx}, "
        "\"happy\":   {\"desc\":\"...\",\"arousal\":0.xx,\"intensity\":0.xx}, "
        "\"sad\":     {\"desc\":\"...\",\"arousal\":0.xx,\"intensity\":0.xx}"
        "}\n"
        f"Seed hint (may ignore): {seed}"
    )

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=520,
    )
    content = resp.choices[0].message.content or ""
    obj = extract_json_loose(content)

    text, neu, hap, sad = _parse_controls(obj)
    validate_plastic_text(text)
    return text, neu, hap, sad


# ----------------------------
# Audio generation
# ----------------------------
def is_speech_tts_model(model_name: str) -> bool:
    return model_name.endswith("-tts") or model_name in {"tts-1", "tts-1-hd"}

def synthesize_with_chat_audio(
    client: OpenAI,
    tts_model: str,
    voice: str,
    text: str,
    ctrl: EmotionControl,
    wav_path: Path,
    emotion: str,
) -> None:
    guidance = prosody_guidance(emotion, ctrl.arousal, ctrl.intensity)
    prompt = (
        f"Emotion control (prosody-only): {ctrl.desc}\n"
        f"Arousal/Intensity guidance: {guidance}\n"
        "Important: natural connected speech is allowed, but DO NOT omit any word and DO NOT add any word.\n"
        f"Read the sentence verbatim: {text}"
    )
    resp = client.chat.completions.create(
        model=tts_model,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[
            {"role": "system", "content": "Generate audio only. Speak the sentence verbatim. Never speak instructions or meta text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    audio = getattr(resp.choices[0].message, "audio", None)
    if not audio or not getattr(audio, "data", None):
        raise RuntimeError("No audio data returned (proxy/model may not support chat-audio modalities).")
    save_b64_audio_to_file(audio.data, wav_path)

def synthesize_with_speech_endpoint(
    client: OpenAI,
    tts_model: str,
    voice: str,
    text: str,
    ctrl: EmotionControl,
    wav_path: Path,
    emotion: str,
    speed: Optional[float] = None,
) -> None:
    guidance = prosody_guidance(emotion, ctrl.arousal, ctrl.intensity)
    instructions = (
        f"Emotion control (prosody-only): {ctrl.desc}\n"
        f"Arousal/Intensity guidance: {guidance}\n"
        "Important: natural connected speech is allowed, but DO NOT omit any word and DO NOT add any word.\n"
        "Speak naturally in English. Read the text exactly; do not paraphrase."
    )

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # 优先官方参数 response_format；不支持时再降级
    try:
        kwargs = dict(
            model=tts_model,
            voice=voice,
            input=text,
            response_format="wav",
            instructions=instructions,
        )
        if speed is not None:
            kwargs["speed"] = float(speed)
        audio = client.audio.speech.create(**kwargs)
    except TypeError as e:
        msg = str(e)

        # 某些实现不支持 instructions：去掉后重试
        if "instructions" in msg:
            try:
                kwargs = dict(model=tts_model, voice=voice, input=text, response_format="wav")
                if speed is not None:
                    kwargs["speed"] = float(speed)
                audio = client.audio.speech.create(**kwargs)
            except TypeError:
                # 兜底：旧实现可能用 format
                kwargs = dict(model=tts_model, voice=voice, input=text, format="wav")
                if speed is not None:
                    kwargs["speed"] = float(speed)
                audio = client.audio.speech.create(**kwargs)
        else:
            kwargs = dict(model=tts_model, voice=voice, input=text, format="wav", instructions=instructions)
            if speed is not None:
                kwargs["speed"] = float(speed)
            audio = client.audio.speech.create(**kwargs)

    # 兼容不同返回类型
    if hasattr(audio, "write_to_file"):
        audio.write_to_file(str(wav_path))
    elif hasattr(audio, "stream_to_file"):
        audio.stream_to_file(str(wav_path))
    elif isinstance(audio, (bytes, bytearray)):
        wav_path.write_bytes(bytes(audio))
    else:
        wav_path.write_bytes(bytes(audio))

def transcribe(client: OpenAI, asr_model: str, wav_path: Path) -> str:
    with wav_path.open("rb") as f:
        out = client.audio.transcriptions.create(
            model=asr_model,
            file=f,
            response_format="text",
        )
    if isinstance(out, str):
        return out.strip()
    return (getattr(out, "text", None) or str(out)).strip()


# ----------------------------
# Request wrapper with controlled retries
# ----------------------------
def call_with_controlled_retries(
    fn,
    *,
    what: str,
    max_tries: int,
    base_backoff: float,
    retry_exceptions: Tuple[type, ...] = (),
):
    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            log(f"ERROR during {what} attempt={attempt}/{max_tries}: {repr(e)}")

            if retry_exceptions and isinstance(e, retry_exceptions):
                sleep_backoff(max(0.2, base_backoff), attempt, cap=2.0)
                continue

            if is_rate_limit_error(e):
                sleep_backoff(base_backoff, attempt, cap=12.0)
                continue
            if is_transient_error(e):
                sleep_backoff(base_backoff, attempt, cap=6.0)
                continue
            raise
    raise RuntimeError(f"Failed {what} after {max_tries} tries. Last error: {repr(last_err)}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=100)

    ap.add_argument("--llm_model", type=str, default="gpt-4o")
    ap.add_argument("--tts_model", type=str, default="gpt-4o-mini-tts")
    ap.add_argument("--asr_model", type=str, default="gpt-4o-mini-transcribe")

    ap.add_argument("--voices", type=str, default=",".join(DEFAULT_VOICES))

    # 过滤策略：neutral 更严；happy/sad 更宽，但默认严格“不丢词”（del=0）
    ap.add_argument("--wer_threshold", type=float, default=None, help="(兼容旧参数) 若设置，则三种情绪都用同一阈值。")
    ap.add_argument("--wer_neutral", type=float, default=0.08)
    ap.add_argument("--wer_emotion", type=float, default=0.16)
    ap.add_argument("--max_del_neutral", type=int, default=0)
    ap.add_argument("--max_del_emotion", type=int, default=0)

    # 合成阶段可选：给情绪更强一点的语速微调（默认不动）
    ap.add_argument("--speed_neutral", type=float, default=None)
    ap.add_argument("--speed_happy", type=float, default=None)
    ap.add_argument("--speed_sad", type=float, default=None)

    ap.add_argument("--max_task_retries", type=int, default=3, help="Per-step retries (LLM/TTS/ASR).")
    ap.add_argument("--backoff_s", type=float, default=0.6, help="Base backoff seconds for 429/5xx.")

    ap.add_argument("--api_base", type=str, default=os.getenv("CLOSEAI_API_BASE", ""))
    ap.add_argument("--api_key", type=str, default=os.getenv("CLOSEAI_API_KEY", ""))

    ap.add_argument("--sdk_max_retries", type=int, default=0)
    ap.add_argument("--timeout_connect", type=float, default=10.0)
    ap.add_argument("--timeout_read", type=float, default=60.0)
    ap.add_argument("--timeout_write", type=float, default=60.0)
    ap.add_argument("--timeout_pool", type=float, default=10.0)

    ap.add_argument("--keep_failed_wav", action="store_true", help="Keep wav files even if filter fails.")
    ap.add_argument("--fail_fast", action="store_true", help="Stop immediately on an unrecoverable error.")

    args = ap.parse_args()

    api_base = normalize_api_base(args.api_base)
    api_key = args.api_key
    if not api_base or not api_key:
        raise ValueError("api_base/api_key 不能为空：请传 --api_base/--api_key 或设置环境变量 CLOSEAI_API_BASE/CLOSEAI_API_KEY")

    timeout = httpx.Timeout(
        connect=args.timeout_connect,
        read=args.timeout_read,
        write=args.timeout_write,
        pool=args.timeout_pool,
    )
    http_client = httpx.Client(timeout=timeout)

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=timeout,
        max_retries=args.sdk_max_retries,
        http_client=http_client,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    if not voices:
        raise ValueError("voices 为空。")

    # resume
    done = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            done = sum(1 for _ in f)

    # thresholds
    if args.wer_threshold is not None:
        wer_neu_th = float(args.wer_threshold)
        wer_emo_th = float(args.wer_threshold)
    else:
        wer_neu_th = float(args.wer_neutral)
        wer_emo_th = float(args.wer_emotion)

    log(f"api_base={api_base}")
    log(f"resume from {done}, target={args.num_samples}")
    log(f"tts_mode={'speech_endpoint' if is_speech_tts_model(args.tts_model) else 'chat_audio'}")
    log(f"sdk_max_retries={args.sdk_max_retries}, timeout={timeout}")
    log(f"WER thresholds: neutral={wer_neu_th}, emotion={wer_emo_th} | max_del neutral={args.max_del_neutral}, emotion={args.max_del_emotion}")

    def _emotion_thresholds(emotion: str) -> Tuple[float, int]:
        if (emotion or "").lower() == "neutral":
            return wer_neu_th, int(args.max_del_neutral)
        return wer_emo_th, int(args.max_del_emotion)

    def _emotion_speed(emotion: str) -> Optional[float]:
        e = (emotion or "").lower()
        if e == "neutral":
            return args.speed_neutral
        if e == "happy":
            return args.speed_happy
        if e == "sad":
            return args.speed_sad
        return None

    for idx in range(done, args.num_samples):
        # uid = f"{idx:06d}"
        uid = f"{100000 + idx:06d}"
        style = TEXT_STYLES[idx % len(TEXT_STYLES)]
        voice = voices[idx % len(voices)]

        try:
            # Step1: emotion-plastic text + controls
            log(f"[{uid}] step1: gen_text_and_controls (style={style}, voice={voice})")
            text, ctrl_neu, ctrl_hap, ctrl_sad = call_with_controlled_retries(
                lambda: gen_text_and_controls(client, args.llm_model, style, seed=idx),
                what=f"{uid}/gen_text_and_controls",
                max_tries=args.max_task_retries,
                base_backoff=args.backoff_s,
                retry_exceptions=(ValueError, KeyError, json.JSONDecodeError),
            )

            wav_neu = out_dir / "wav" / "neutral" / f"{uid}_{voice}.wav"
            wav_hap = out_dir / "wav" / "happy" / f"{uid}_{voice}.wav"
            wav_sad = out_dir / "wav" / "sad" / f"{uid}_{voice}.wav"

            def synth_and_asr(emotion: str, ctrl: EmotionControl, wav_path: Path) -> Tuple[str, float, int, int, int]:
                th, max_del = _emotion_thresholds(emotion)
                speed = _emotion_speed(emotion)

                log(f"[{uid}] step2: synth+asr {emotion} (th={th}, max_del={max_del})")

                def _do():
                    # synth
                    if is_speech_tts_model(args.tts_model):
                        synthesize_with_speech_endpoint(
                            client, args.tts_model, voice, text, ctrl, wav_path, emotion=emotion, speed=speed
                        )
                    else:
                        synthesize_with_chat_audio(
                            client, args.tts_model, voice, text, ctrl, wav_path, emotion=emotion
                        )

                    # asr + stats
                    hyp = transcribe(client, args.asr_model, wav_path)
                    w, ins, dele, sub, _ = wer_stats(text, hyp)

                    # 关键：情绪更强允许更自然连读 => 放宽 WER，但不允许丢词（dele=0 默认）
                    if dele > max_del or w > th:
                        raise ValueError(
                            f"filter_fail emotion={emotion} WER={w:.3f} (th={th}) ins/del/sub={ins}/{dele}/{sub} (max_del={max_del})"
                        )
                    return hyp, w, ins, dele, sub

                return call_with_controlled_retries(
                    _do,
                    what=f"{uid}/synth_asr_{emotion}",
                    max_tries=args.max_task_retries,
                    base_backoff=args.backoff_s,
                    retry_exceptions=(ValueError,),
                )

            asr_neu, wer_neu, ins_neu, del_neu, sub_neu = synth_and_asr("neutral", ctrl_neu, wav_neu)
            asr_hap, wer_hap, ins_hap, del_hap, sub_hap = synth_and_asr("happy",   ctrl_hap, wav_hap)
            asr_sad, wer_sad, ins_sad, del_sad, sub_sad = synth_and_asr("sad",     ctrl_sad, wav_sad)

            # kept 判定：按上面的 synth_and_asr 已经保证达标；到这里基本都是 kept=True
            kept = True

            item = {
                "id": uid,
                "style": style,
                "voice": voice,
                "text": text,

                "neutral": {"desc": ctrl_neu.desc, "arousal": ctrl_neu.arousal, "intensity": ctrl_neu.intensity},
                "happy":   {"desc": ctrl_hap.desc, "arousal": ctrl_hap.arousal, "intensity": ctrl_hap.intensity},
                "sad":     {"desc": ctrl_sad.desc, "arousal": ctrl_sad.arousal, "intensity": ctrl_sad.intensity},

                "wav_neutral": str(wav_neu.relative_to(out_dir)),
                "wav_happy":   str(wav_hap.relative_to(out_dir)),
                "wav_sad":     str(wav_sad.relative_to(out_dir)),

                "asr_neutral": asr_neu,
                "asr_happy":   asr_hap,
                "asr_sad":     asr_sad,

                "wer_neutral": wer_neu, "ins_neutral": ins_neu, "del_neutral": del_neu, "sub_neutral": sub_neu,
                "wer_happy":   wer_hap, "ins_happy":   ins_hap, "del_happy":   del_hap, "sub_happy":   sub_hap,
                "wer_sad":     wer_sad, "ins_sad":     ins_sad, "del_sad":     del_sad, "sub_sad":     sub_sad,

                "kept": kept,
                "api_base": api_base,
                "models": {"llm": args.llm_model, "tts": args.tts_model, "asr": args.asr_model},

                "filter_policy": {
                    "wer_threshold": args.wer_threshold,
                    "wer_neutral": wer_neu_th,
                    "wer_emotion": wer_emo_th,
                    "max_del_neutral": int(args.max_del_neutral),
                    "max_del_emotion": int(args.max_del_emotion),
                },
            }

            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            log(
                f"[{uid}] kept={kept} | "
                f"WER n/h/s={wer_neu:.3f}/{wer_hap:.3f}/{wer_sad:.3f} | "
                f"DEL n/h/s={del_neu}/{del_hap}/{del_sad}"
            )

        except Exception as e:
            log(f"[{uid}] FATAL ERROR: {repr(e)}")
            if not args.keep_failed_wav:
                # 若失败，尽量清掉残留 wav（避免你以为是有效样本）
                safe_unlink(out_dir / "wav" / "neutral" / f"{uid}_{voice}.wav")
                safe_unlink(out_dir / "wav" / "happy" / f"{uid}_{voice}.wav")
                safe_unlink(out_dir / "wav" / "sad" / f"{uid}_{voice}.wav")

            if args.fail_fast:
                raise
            continue


if __name__ == "__main__":
    main()





