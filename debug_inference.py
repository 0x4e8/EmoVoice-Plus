# import os
# import sys
# import runpy
# import shlex

# # =========================
# # 1) 固定工程路径 & 环境变量
# # =========================
# repo_root = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice"

# # 等价于你 bash 里的 export PYTHONPATH=.../src
# sys.path.append(repo_root)
# sys.path.append(os.path.join(repo_root, "src"))
# sys.path.append(os.path.join(repo_root, "examples", "tts"))

# # 确保相对路径、Hydra 输出目录等行为稳定
# os.chdir(repo_root)

# # 对齐你 bash 里的环境变量（按需保留/删减）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # =========================
# # 2) 组装你 bash 的推理参数（原样写进 sys.argv）
# # =========================
# llm_path = f"{repo_root}/ckpt/ckpts/Qwen/Qwen2.5-0.5B"
# codec_decoder_path = f"{repo_root}/ckpt/ckpts/CosyVoice/CosyVoice-300M-SFT"
# phn_tokenizer = f"{repo_root}/ckpt/Qwen2.5-0.5B-phn"
# ckpt_root = f"{repo_root}/ckpt"

# val_data_path = f"{repo_root}/EmoVoice-DB/test_mini.jsonl"
# tmp_ckpt_path = f"{repo_root}/debug2/tts_epoch_10_step_2631/model.pt"

# # 你 bash 里的 decode_log 变量用了未定义 repetition_penalty/dataset_sample_seed，
# # 这里给一个确定的目录名，便于 debug。
# decode_log = f"{ckpt_root}/tts_decode_test_debug_epoch10_step2631"

# # 目标脚本（保持与 bash 一致）
# target_script = "examples/tts/inference_tts.py"

# args = f"""
# python {target_script}
# hydra.run.dir={ckpt_root}
# ++model_config.llm_name=qwen2.5-0.5b
# ++model_config.llm_path={llm_path}
# ++model_config.llm_dim=896
# ++model_config.codec_decoder_path={codec_decoder_path}
# ++model_config.codec_decode=true
# ++model_config.vocab_config.code_layer=3
# ++model_config.vocab_config.total_audio_vocabsize=4160
# ++model_config.vocab_config.total_vocabsize=156160
# ++model_config.codec_decoder_type=CosyVoice
# ++model_config.group_decode=true
# ++model_config.group_decode_adapter_type=linear
# ++model_config.phn_tokenizer={phn_tokenizer}
# ++dataset_config.dataset=speech_dataset_tts
# ++dataset_config.val_data_path={val_data_path}
# ++dataset_config.train_data_path={val_data_path}
# ++dataset_config.inference_mode=true
# ++dataset_config.vocab_config.code_layer=3
# ++dataset_config.vocab_config.total_audio_vocabsize=4160
# ++dataset_config.vocab_config.total_vocabsize=156160
# ++dataset_config.num_latency_tokens=5
# ++dataset_config.do_layershift=false
# ++dataset_config.use_emo=true
# ++train_config.model_name=tts
# ++train_config.freeze_encoder=true
# ++train_config.freeze_llm=true
# ++train_config.freeze_group_decode_adapter=true
# ++train_config.batching_strategy=custom
# ++train_config.num_epochs=1
# ++train_config.val_batch_size=1
# ++train_config.num_workers_dataloader=2
# ++decode_config.text_repetition_penalty=1.2
# ++decode_config.audio_repetition_penalty=1.2
# ++decode_config.max_new_tokens=3000
# ++decode_config.do_sample=false
# ++decode_config.top_p=1.0
# ++decode_config.top_k=0
# ++decode_config.temperature=1.0
# ++decode_config.decode_text_only=false
# ++decode_config.num_latency_tokens=5
# ++decode_config.do_layershift=false
# ++decode_log={decode_log}
# ++ckpt_path={tmp_ckpt_path}
# ++output_text_only=false
# ++speech_sample_rate=22050
# ++log_config.log_file={decode_log}/infer.log
# """.strip()

# argv = shlex.split(args)

# # =========================
# # 3) 模拟命令行：python xxx.py <args...>
# # =========================
# if argv[0] == "python":
#     argv.pop(0)

# # 关键：覆盖 sys.argv，让 hydra/argparse 正常解析
# sys.argv = [argv[0]] + argv[1:]

# # =========================
# # 4) runpy 跑原脚本（VSCode 下断点即可）
# # =========================
# runpy.run_path(argv[0], run_name="__main__")
import os
import sys
import runpy
import shlex

# =========================
# 1) 固定工程路径 & 环境变量
# =========================
repo_root = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice"
code_dir = "examples/tts"
target_script = f"{code_dir}/inference_tts.py"

# 等价于：export PYTHONPATH=.../src
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "examples", "tts"))

# 保证相对路径 / hydra 行为稳定
os.chdir(repo_root)

# 对齐 bash export
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# =========================
# 2) 组装你的最新推理参数
# =========================
llm_path = f"{repo_root}/ckpt/ckpts/Qwen/Qwen2.5-0.5B"
codec_decoder_path = f"{repo_root}/ckpt/ckpts/CosyVoice/CosyVoice-300M-SFT"
phn_tokenizer = f"{repo_root}/ckpt/Qwen2.5-0.5B-phn"
ckpt_path = f"{repo_root}/ckpt"
split = "test"
val_data_path = f"{repo_root}/EmoVoice-DB/test_mini_happy.jsonl"

# vocabulary
code_layer = 3
total_audio_vocabsize = 4160
total_vocabsize = 156160

# code settings
codec_decoder_type = "CosyVoice"
num_latency_tokens = 5
do_layershift = "false"  # hydra bool: "true"/"false"

# model settings
group_decode = "true"
group_decode_adapter_type = "linear"

# decode config
text_repetition_penalty = 1.2
audio_repetition_penalty = 1.2
max_new_tokens = 3000
do_sample = "false"
top_p = 1.0
top_k = 0
temperature = 1.0
decode_text_only = "false"
output_text_only = "false"
speech_sample_rate = 22050

tmp_ckpt_path = f"{repo_root}/debug2/tts_epoch_10_step_2631/model.pt"

# 你 bash 的 decode_log 里有未定义的 repetition_penalty / dataset_sample_seed
# 这里给一个确定且可读的名字，建议与你原意保持一致
decode_log = f"{ckpt_path}/tts_decode_{split}_greedy_kaiyuan_epoch_10-happy-debug"
# 如果你希望把 text/audio repetition penalty 信息也带上，可用：
# decode_log = f"{ckpt_path}/tts_decode_{split}_textrp{text_repetition_penalty}_audiorp{audio_repetition_penalty}_greedy_kaiyuan_epoch_10-happy-T"

# emosteer
emosteer_enable = "true"
emosteer_path = f"{repo_root}/tools/happy_steering_activations_10000.pt"
emosteer_strength = 0.8
emosteer_num_steps = 32

# 确保输出目录存在（避免某些写 log 的地方报错）
os.makedirs(decode_log, exist_ok=True)

# =========================
# 3) 拼出等价命令并塞进 sys.argv
# =========================
args = f"""
python {target_script}
hydra.run.dir={ckpt_path}
++model_config.llm_name=qwen2.5-0.5b
++model_config.llm_path={llm_path}
++model_config.llm_dim=896
++model_config.codec_decoder_path={codec_decoder_path}
++model_config.codec_decode=true
++model_config.vocab_config.code_layer={code_layer}
++model_config.vocab_config.total_audio_vocabsize={total_audio_vocabsize}
++model_config.vocab_config.total_vocabsize={total_vocabsize}
++model_config.codec_decoder_type={codec_decoder_type}
++model_config.group_decode={group_decode}
++model_config.group_decode_adapter_type={group_decode_adapter_type}
++model_config.phn_tokenizer={phn_tokenizer}
++dataset_config.dataset=speech_dataset_tts
++dataset_config.val_data_path={val_data_path}
++dataset_config.train_data_path={val_data_path}
++dataset_config.inference_mode=true
++dataset_config.vocab_config.code_layer={code_layer}
++dataset_config.vocab_config.total_audio_vocabsize={total_audio_vocabsize}
++dataset_config.vocab_config.total_vocabsize={total_vocabsize}
++dataset_config.num_latency_tokens={num_latency_tokens}
++dataset_config.do_layershift={do_layershift}
++dataset_config.use_emo=true
++train_config.model_name=tts
++train_config.freeze_encoder=true
++train_config.freeze_llm=true
++train_config.freeze_group_decode_adapter=true
++train_config.batching_strategy=custom
++train_config.num_epochs=1
++train_config.val_batch_size=1
++train_config.num_workers_dataloader=2
++decode_config.text_repetition_penalty={text_repetition_penalty}
++decode_config.audio_repetition_penalty={audio_repetition_penalty}
++decode_config.max_new_tokens={max_new_tokens}
++decode_config.do_sample={do_sample}
++decode_config.top_p={top_p}
++decode_config.top_k={top_k}
++decode_config.temperature={temperature}
++decode_config.decode_text_only={decode_text_only}
++decode_config.num_latency_tokens={num_latency_tokens}
++decode_config.do_layershift={do_layershift}
++decode_log={decode_log}
++ckpt_path={tmp_ckpt_path}
++output_text_only={output_text_only}
++speech_sample_rate={speech_sample_rate}
++log_config.log_file={decode_log}/infer.log
++decode_config.emosteer.enable={emosteer_enable}
++decode_config.emosteer.steering_path={emosteer_path}
++decode_config.emosteer.steering_strength={emosteer_strength}
++decode_config.emosteer.num_steps={emosteer_num_steps}
""".strip()

argv = shlex.split(args)

if argv[0] == "python":
    argv.pop(0)

# 覆盖 sys.argv，让 hydra 正常解析
sys.argv = [argv[0]] + argv[1:]

# =========================
# 4) runpy 跑原脚本（VSCode 下断点即可）
# =========================
runpy.run_path(argv[0], run_name="__main__")
