# 添加代码
# examples/tts/utils/build_emosteer_vecs.py
import argparse, torch, torchaudio
from utils.codec_utils import setup_codec
from utils.emosteer import ActivationRecorder, save_vecs
from utils.cosyvoice.utils.file_utils import load_wav

def extract_prompt(codec_decoder, prompt_wav_path, cosy_ver: int):
    prompt_16k = load_wav(prompt_wav_path, 16000)
    prompt_token, prompt_token_len = codec_decoder.frontend._extract_speech_token(prompt_16k)
    if cosy_ver == 1:
        prompt_res = torchaudio.transforms.Resample(16000, 22050)(prompt_16k)
    else:
        prompt_res = torchaudio.transforms.Resample(16000, 24000)(prompt_16k)
    prompt_feat, prompt_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_res)
    emb = codec_decoder.frontend._extract_spk_embedding(prompt_16k)
    return prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, emb

@torch.no_grad()
def run_one(codec_decoder, wav_path, prompt_pack):
    speech_16k = load_wav(wav_path, 16000)
    token, token_len = codec_decoder.frontend._extract_speech_token(speech_16k)
    prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, emb = prompt_pack

    # 只跑 flow（不跑 hift）
    mel, _ = codec_decoder.model.flow.inference(
        token=token, token_len=token_len,
        prompt_token=prompt_token, prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat, prompt_feat_len=prompt_feat_len,
        embedding=emb,
        flow_cache=torch.zeros(1, 80, 0, 2, device=codec_decoder.model.flow.device if hasattr(codec_decoder.model.flow, "device") else token.device)
    )
    return mel

def read_list(path):
    return [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_dir", required=True)
    ap.add_argument("--cosy_ver", type=int, default=1)
    ap.add_argument("--prompt_wav", required=True)
    ap.add_argument("--neutral_list", required=True)
    ap.add_argument("--emotion_list", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    class Dummy: pass
    train_cfg = Dummy(); train_cfg.enable_fsdp=False; train_cfg.enable_ddp=False
    model_cfg = Dummy()
    model_cfg.codec_decoder_path=args.codec_dir
    model_cfg.cosyvoice_version=args.cosy_ver
    model_cfg.codec_decoder_type="cosyvoice"
    model_cfg.save_audio_token=False

    codec = setup_codec(train_cfg, model_cfg)
    estimator = codec.model.flow.decoder.estimator

    prompt_pack = extract_prompt(codec, args.prompt_wav, args.cosy_ver)

    def collect(list_path):
        rec = ActivationRecorder(every_n_calls=1)
        with rec.attach(estimator):
            for wav in read_list(list_path):
                run_one(codec, wav, prompt_pack)
        return rec.mean_vecs()

    neu = collect(args.neutral_list)
    emo = collect(args.emotion_list)
    vecs = []
    for a, b in zip(emo, neu):
        vecs.append(None if (a is None or b is None) else (a - b))
    save_vecs(args.out, vecs, meta={"cosy_ver": args.cosy_ver})
    print("saved:", args.out)

if __name__ == "__main__":
    main()
