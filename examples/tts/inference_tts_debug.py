# examples/tts/inference_tts_debug.py
import os
import sys
import logging
from omegaconf import OmegaConf

# 你原来的入口：generate_tts_batch.main(cfg)
from generate_tts_batch import main as inference
from tts_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, DecodeConfig
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)

    debug: bool = False
    metric: str = "acc"

    decode_log: str = "output/decode_log"
    ckpt_path: Optional[str] = None
    peft_ckpt: Optional[str] = None

    output_text_only: bool = False
    speech_sample_rate: int = 24000

    audio_prompt_path: Optional[str] = None
    multi_round: bool = False


def _set_env():
    # 对齐你 bash 里的环境变量（VSCode debug 直接运行也生效）
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_cfg():
    # ====== 你的工程根目录（按需改）======
    repo_root = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice"

    # 保证 VSCode 里直接跑也能 import
    sys.path.append(os.path.join(repo_root, "src"))
    sys.path.append(os.path.join(repo_root, "examples", "tts"))

    # ====== 对齐 bash 参数 ======
    llm_path = os.path.join(repo_root, "ckpt/ckpts/Qwen/Qwen2.5-0.5B")
    codec_decoder_path = os.path.join(repo_root, "ckpt/ckpts/CosyVoice/CosyVoice-300M-SFT")
    phn_tokenizer = os.path.join(repo_root, "ckpt/Qwen2.5-0.5B-phn")

    ckpt_root = os.path.join(repo_root, "ckpt")
    split = "test"
    val_data_path = os.path.join(repo_root, "EmoVoice-DB/test_mini.jsonl")

    code_layer = 3
    total_audio_vocabsize = 4160
    total_vocabsize = 156160

    codec_decoder_type = "CosyVoice"
    num_latency_tokens = 5
    do_layershift = False

    group_decode = True
    group_decode_adapter_type = "linear"

    text_repetition_penalty = 1.2
    audio_repetition_penalty = 1.2
    max_new_tokens = 3000
    do_sample = False
    top_p = 1.0
    top_k = 0
    temperature = 1.0
    decode_text_only = False
    output_text_only = False
    speech_sample_rate = 22050

    tmp_ckpt_path = os.path.join(
        repo_root, "debug2/tts_epoch_10_step_2631/model.pt"
    )

    # 你 bash 里 decode_log 用了未定义变量 repetition_penalty/dataset_sample_seed，
    # 这里直接生成一个确定的目录名，便于 debug。
    decode_log = os.path.join(
        ckpt_root,
        f"tts_decode_{split}_textrp{text_repetition_penalty}_audiorp{audio_repetition_penalty}"
        f"_greedy_debug_epoch10_step2631"
    )
    os.makedirs(decode_log, exist_ok=True)

    # ====== 组装 cfg（不走 hydra 命令行）======
    cfg = OmegaConf.structured(RunConfig())
    OmegaConf.set_struct(cfg, False)  # 允许加新字段（比如 hydra.run.dir）

    # 模拟 hydra.run.dir（有些代码会读）
    cfg.hydra = {"run": {"dir": ckpt_root}}

    # model_config
    cfg.model_config.llm_name = "qwen2.5-0.5b"
    cfg.model_config.llm_path = llm_path
    cfg.model_config.llm_dim = 896
    cfg.model_config.codec_decoder_path = codec_decoder_path
    cfg.model_config.codec_decode = True
    cfg.model_config.vocab_config.code_layer = code_layer
    cfg.model_config.vocab_config.total_audio_vocabsize = total_audio_vocabsize
    cfg.model_config.vocab_config.total_vocabsize = total_vocabsize
    cfg.model_config.codec_decoder_type = codec_decoder_type
    cfg.model_config.group_decode = group_decode
    cfg.model_config.group_decode_adapter_type = group_decode_adapter_type
    cfg.model_config.phn_tokenizer = phn_tokenizer

    # dataset_config
    cfg.dataset_config.dataset = "speech_dataset_tts"
    cfg.dataset_config.val_data_path = val_data_path
    cfg.dataset_config.train_data_path = val_data_path
    cfg.dataset_config.inference_mode = True
    cfg.dataset_config.vocab_config.code_layer = code_layer
    cfg.dataset_config.vocab_config.total_audio_vocabsize = total_audio_vocabsize
    cfg.dataset_config.vocab_config.total_vocabsize = total_vocabsize
    cfg.dataset_config.num_latency_tokens = num_latency_tokens
    cfg.dataset_config.do_layershift = do_layershift
    cfg.dataset_config.use_emo = True

    # train_config（推理时很多字段只是被沿用，但按你 bash 对齐）
    cfg.train_config.model_name = "tts"
    cfg.train_config.freeze_encoder = True
    cfg.train_config.freeze_llm = True
    cfg.train_config.freeze_group_decode_adapter = True
    cfg.train_config.batching_strategy = "custom"
    cfg.train_config.num_epochs = 1
    cfg.train_config.val_batch_size = 1
    cfg.train_config.num_workers_dataloader = 2

    # decode_config
    cfg.decode_config.text_repetition_penalty = text_repetition_penalty
    cfg.decode_config.audio_repetition_penalty = audio_repetition_penalty
    cfg.decode_config.max_new_tokens = max_new_tokens
    cfg.decode_config.do_sample = do_sample
    cfg.decode_config.top_p = top_p
    cfg.decode_config.top_k = top_k
    cfg.decode_config.temperature = temperature
    cfg.decode_config.decode_text_only = decode_text_only
    cfg.decode_config.num_latency_tokens = num_latency_tokens
    cfg.decode_config.do_layershift = do_layershift

    # 顶层字段
    cfg.decode_log = decode_log
    cfg.ckpt_path = tmp_ckpt_path
    cfg.output_text_only = output_text_only
    cfg.speech_sample_rate = speech_sample_rate

    # log_config
    cfg.log_config.log_file = os.path.join(decode_log, "infer.log")

    # 可选：打开 pdb（不需要的话保持 False，用 VSCode 断点即可）
    cfg.debug = False

    return cfg


def main():
    _set_env()
    cfg = build_cfg()

    log_level = getattr(logging, "INFO")
    logging.basicConfig(level=log_level)

    # 如果你想看最终 cfg 长什么样，可以取消注释
    # print(OmegaConf.to_yaml(cfg))

    if cfg.get("debug", False):
        import pdb
        pdb.set_trace()

    inference(cfg)


if __name__ == "__main__":
    main()
