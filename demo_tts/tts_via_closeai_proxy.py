# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# tts_emotion_parametric.py (via CloseAI proxy, English->English WAV)

# Run:
#   python tts_emotion_parametric.py

# Deps:
#   pip install -U openai
# """

# from __future__ import annotations
# import os
# import sys
# from pathlib import Path
# from openai import OpenAI


# # =========================
# # 你只需要改这里的参数
# # =========================

# # ---- CloseAI / OpenAI-compatible proxy ----
# # CloseAI 文档示例：base_url='https://api.openai-proxy.org/v1' 并配套它们的 api_key
# API_BASE = "https://api.openai-proxy.org/v1"
# API_KEY  = "sk-0aEq4mIVmc8XWu8mHnGvlq33BE5uIYHxTToR9DA5cTnLv8jj"  # 填你的 CloseAI Key；不想写死就留空并用环境变量 OPENAI_API_KEY

# MODEL = "gpt-4o-mini-tts"
# VOICE = "marin"

# # ✅ English input text
# TEXT = "I got a good grade in the speech information processing course!"

# # Emotion category: neutral / happy / sad / angry / fear / surprise / disgust
# EMOTION = "sad"

# # intensity: 0~1
# INTENSITY = 0.70

# # arousal: 0~1
# AROUSAL = 0.20

# # auto speed derived from arousal/intensity (more stable)
# USE_AUTO_SPEED = True

# # if USE_AUTO_SPEED=False, use this fixed speed (0.25~4.0)
# MANUAL_SPEED = 1.00

# # extra notes (optional): e.g. "whispery", "more formal", "British accent"
# EXTRA_NOTES = ""

# OUT_WAV = "out_parametric_en_sad.wav"

# # =========================


# def _clamp(x: float, lo: float, hi: float) -> float:
#     return max(lo, min(hi, x))


# def _clamp01(x: float) -> float:
#     return _clamp(float(x), 0.0, 1.0)


# def auto_speed(arousal: float, intensity: float) -> float:
#     """
#     Conservative speed:
#       - higher arousal -> faster
#       - higher intensity -> slightly faster
#     """
#     a = _clamp01(arousal)
#     i = _clamp01(intensity)
#     s = 0.80 + 0.70 * a          # 0.80 ~ 1.50
#     s *= (0.92 + 0.16 * i)       # 0.92 ~ 1.08
#     return _clamp(s, 0.25, 4.0)


# def build_instructions(emotion: str, intensity: float, arousal: float, extra_notes: str = "") -> str:
#     """
#     Map (emotion + intensity + arousal) -> stable, executable English instructions.
#     """
#     emo = (emotion or "neutral").strip().lower()
#     i = _clamp01(intensity)
#     a = _clamp01(arousal)

#     # Hard constraints => stability (no paraphrase/translation)
#     common = [
#         "You are a professional English voice actor.",
#         "Language: English (US).",
#         "Read the input VERBATIM. Do NOT translate. Do NOT add, remove, or change any words.",
#         "Clear articulation, natural pauses, not theatrical.",
#         f"Emotion={emo}; intensity={i:.2f}; arousal={a:.2f}.",
#     ]

#     pace = "slightly slower" if a < 0.33 else ("normal pace" if a < 0.66 else "slightly faster")
#     energy = "low energy" if a < 0.33 else ("medium energy" if a < 0.66 else "high energy")
#     emphasis = "subtle emphasis" if i < 0.33 else ("noticeable emphasis" if i < 0.66 else "strong emphasis")

#     if emo == "neutral":
#         style = [
#             "Tone: neutral, restrained.",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: mid, small variation; natural intonation.",
#             "Timbre: clean, not breathy.",
#         ]
#     elif emo == "happy":
#         style = [
#             "Tone: positive and warm, not shouting.",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: slightly higher; brighter intonation; light uplift at ends.",
#             "Hint of a smile, but keep clarity.",
#         ]
#     elif emo == "sad":
#         style = [
#             "Tone: restrained sadness (not crying).",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: slightly lower; gentle downward endings.",
#             "Timbre: slightly breathy is OK, but keep words clear.",
#         ]
#     elif emo == "angry":
#         style = [
#             "Tone: anger / irritation, controlled (do not lose intelligibility).",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: tighter / slightly higher; harder consonants; shorter stress.",
#             "Pauses: crisp and deliberate.",
#         ]
#     elif emo == "fear":
#         style = [
#             "Tone: anxious / fearful, cautious.",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: slightly higher but shaky; mild hesitation pauses (no extra words).",
#             "Breathing: slightly audible but not disruptive.",
#         ]
#     elif emo == "surprise":
#         style = [
#             "Tone: surprise / disbelief, quick lift on key words.",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: higher at the start then return to natural.",
#             "Bright but still natural.",
#         ]
#     elif emo == "disgust":
#         style = [
#             "Tone: disgust / disdain, cool and detached.",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Pitch: flatter; slight lowering on key words.",
#             "Pauses: slightly distancing but not long.",
#         ]
#     else:
#         style = [
#             f"Tone: {emo} (custom).",
#             f"Pace: {pace}; Energy: {energy}; Emphasis: {emphasis}.",
#             "Keep it natural, clear, and verbatim.",
#         ]

#     fine = []
#     if i >= 0.66:
#         fine.append("High intensity: make the emotional coloring more obvious, but avoid exaggeration.")
#     elif i <= 0.33:
#         fine.append("Low intensity: keep emotional coloring subtle and restrained.")

#     if a <= 0.33:
#         fine.append("Low arousal: softer volume and gentler pacing; avoid rushing.")
#     elif a >= 0.66:
#         fine.append("High arousal: more drive and focus; slightly faster pacing; keep clarity.")

#     notes = []
#     extra_notes = (extra_notes or "").strip()
#     if extra_notes:
#         notes.append(f"Extra notes: {extra_notes}")

#     lines = []
#     lines.extend(common)
#     lines.append("Delivery:")
#     for s in style:
#         lines.append(f"- {s}")
#     for f in fine:
#         lines.append(f"- {f}")
#     for n in notes:
#         lines.append(f"- {n}")

#     return "\n".join(lines)


# def main() -> int:
#     # Prefer file API_KEY; fallback to env OPENAI_API_KEY
#     api_key = API_KEY.strip() if isinstance(API_KEY, str) else ""
#     if not api_key:
#         api_key = os.getenv("OPENAI_API_KEY", "").strip()

#     if not api_key:
#         print("ERROR: No API key. Fill API_KEY in the file or set env OPENAI_API_KEY.", file=sys.stderr)
#         return 2

#     instructions = build_instructions(EMOTION, INTENSITY, AROUSAL, EXTRA_NOTES)
#     speed = auto_speed(AROUSAL, INTENSITY) if USE_AUTO_SPEED else float(MANUAL_SPEED)

#     out_path = Path(OUT_WAV).expanduser().resolve()
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # ✅ Key change: use proxy base_url + proxy api_key
#     client = OpenAI(base_url=API_BASE, api_key=api_key)

#     try:
#         with client.audio.speech.with_streaming_response.create(
#             model=MODEL,
#             voice=VOICE,
#             input=TEXT,
#             instructions=instructions,
#             response_format="wav",
#             speed=speed,
#         ) as response:
#             response.stream_to_file(out_path)
#     except Exception as e:
#         print(f"ERROR: TTS failed: {type(e).__name__}: {e}", file=sys.stderr)
#         print("Hints:", file=sys.stderr)
#         print("1) Ensure API_BASE ends with /v1.", file=sys.stderr)
#         print("2) Ensure you are using the CloseAI key that matches this API_BASE.", file=sys.stderr)
#         return 1

#     print("[OK] WAV saved:", out_path)
#     print("[INFO] speed =", speed)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English text -> English WAV via OpenAI-compatible proxy (CloseAI).

Run:
  python tts_en_via_proxy.py

Deps:
  pip install -U openai
"""

from __future__ import annotations
from pathlib import Path
from openai import OpenAI

# =========================
# EDIT THESE
# =========================
API_BASE = "https://api.openai-proxy.org/v1"   # your proxy base (must include /v1)
API_KEY  = "sk-0aEq4mIVmc8XWu8mHnGvlq33BE5uIYHxTToR9DA5cTnLv8jj"       # your proxy key

MODEL = "gpt-4o-mini-tts"
# Voices supported include: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse, marin, cedar :contentReference[oaicite:2]{index=2}
# For best quality, OpenAI recommends marin or cedar in their voice docs. :contentReference[oaicite:3]{index=3}
VOICE = "ash"

SPEED = 1.0   # 0.25 ~ 4.0 :contentReference[oaicite:4]{index=4}

TEXT = "I got EXCELLENT grades in my speech signal processing course, which MAKES me very EXCITED!"

happy = "Depicting a serene and happy moment with a soft, soothing tone."
angry = "controlled anger, high arousal, firm and sharp delivery, tighter/lower patience tone, clipped pacing, stronger emphasis on key words, clear articulation (no shouting)."
neutral = "Evoking a neutral consistency through the habitual sounds of the city."
# IMPORTANT: use English instructions for English output
INSTRUCTIONS = f"""
Read the input VERBATIM. Do not translate or add any extra words.
Style: {happy}
""".strip()
# =========================

OUT_WAV = "out_en_happy.wav"

def main():
    out_path = Path(OUT_WAV).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # response_format supports wav/mp3/opus/aac/flac/pcm :contentReference[oaicite:5]{index=5}
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=TEXT,
        instructions=INSTRUCTIONS,   # controls delivery style :contentReference[oaicite:6]{index=6}
        response_format="wav",
        speed=SPEED,
    ) as response:
        response.stream_to_file(out_path)

    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
