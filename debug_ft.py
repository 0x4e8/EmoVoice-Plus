import os
import sys
import runpy
import shlex

# =========================
# 1) 固定工程路径 & 环境变量
# =========================
repo_root = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice"
code_dir = "examples/tts"
target_script = f"{code_dir}/finetune_tts.py"

# 等价于：export PYTHONPATH=$PYTHONPATH:/.../EmoVoice/src
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "examples", "tts"))

# 保证相对路径、Hydra 输出目录等行为稳定
os.chdir(repo_root)

# 对齐 bash export
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# =========================
# 2) 组装你 bash 里的参数（单卡 debug 路径）
#    说明：你当前 CUDA_VISIBLE_DEVICES=0 => 单卡，不走 torchrun
# =========================
llm_path = f"{repo_root}/ckpt/ckpts/Qwen/Qwen2.5-0.5B"
llm_name = "Qwen2.5-0.5b"
llm_dim = 896
phn_tokenizer = f"{repo_root}/ckpt/Qwen2.5-0.5B-phn"

# vocab
code_layer = 3
total_audio_vocabsize = 4160
llm_vocabsize = 152000
total_vocabsize = total_audio_vocabsize + llm_vocabsize  # 156160

# code settings
num_latency_tokens = 5
do_layershift = "false"  # hydra bool prefer "true"/"false"

# dataset
train_data_path = f"{repo_root}/EmoVoice-DB/train_shuffled_5times.jsonl"
val_data_path = f"{repo_root}/EmoVoice-DB/val.jsonl"

# training
batch_size_training = 16
use_fp16 = "true"
use_peft = "false"
num_epochs = 1
lr = "1e-5"
warmup_steps = 1000
total_steps = 1974

# validation
validation_interval = 1973
split_size = 0.01

# model settings
group_decode = "true"
group_decode_adapter_type = "linear"

# log / exp
exp_name = "debug5"
wandb_entity_name = "0x4e8"
wandb_project_name = "EmoVoice"

home_dir = repo_root
output_dir = f"{home_dir}/{exp_name}"
ckpt_root = f"{repo_root}/ckpt"  # 你 bash 里 ckpt_path 变量（用于 resume/载入）
style_emb_pt = f"{repo_root}/EmoVoice-DB/train_and_val.style_emb.pt"

# 你 bash 的逻辑：exp_name == "debug" 才关 wandb
use_wandb = "false" if exp_name == "debug" else "true"
use_wandb = "false"
wandb_exp_name = exp_name

# 你 bash 里实际传入的是：++ckpt_path=$ckpt_path/0.5b-pp_model.pt
init_ckpt_path = f"{ckpt_root}/0.5b-pp_model.pt"

# =========================
# 3) 拼出等价命令并塞进 sys.argv
# =========================
args = f"""
python {target_script}
hydra.run.dir={output_dir}
++model_config.llm_name={llm_name}
++model_config.llm_path={llm_path}
++model_config.llm_dim={llm_dim}
++model_config.vocab_config.code_layer={code_layer}
++model_config.vocab_config.total_audio_vocabsize={total_audio_vocabsize}
++model_config.vocab_config.total_vocabsize={total_vocabsize}
++model_config.group_decode={group_decode}
++model_config.group_decode_adapter_type={group_decode_adapter_type}
++model_config.phn_tokenizer={phn_tokenizer}

++dataset_config.dataset=speech_dataset_tts
++dataset_config.train_data_path={train_data_path}
++dataset_config.val_data_path={val_data_path}
++dataset_config.seed=42
++dataset_config.split_size={split_size}
++dataset_config.vocab_config.code_layer={code_layer}
++dataset_config.vocab_config.total_audio_vocabsize={total_audio_vocabsize}
++dataset_config.vocab_config.total_vocabsize={total_vocabsize}
++dataset_config.num_latency_tokens={num_latency_tokens}
++dataset_config.do_layershift={do_layershift}
++dataset_config.use_emo=true

++train_config.model_name=tts
++train_config.num_epochs={num_epochs}
++train_config.freeze_encoder=true
++train_config.freeze_llm=false
++train_config.batching_strategy=custom
++train_config.warmup_steps={warmup_steps}
++train_config.total_steps={total_steps}
++train_config.lr={lr}
++train_config.validation_interval={validation_interval}
++train_config.batch_size_training={batch_size_training}
++train_config.val_batch_size={batch_size_training}
++train_config.num_workers_dataloader=0
++train_config.output_dir={output_dir}
++train_config.use_fp16={use_fp16}
++train_config.use_peft={use_peft}

++metric=acc

++log_config.use_wandb={use_wandb}
++log_config.wandb_entity_name={wandb_entity_name}
++log_config.wandb_project_name={wandb_project_name}
++log_config.wandb_exp_name={wandb_exp_name}
++log_config.wandb_dir={output_dir}
++log_config.log_file={output_dir}/exp.log
++log_config.log_interval=100

++ckpt_path={init_ckpt_path}

++train_config.align.enable=true
++train_config.align.style_emb_pt={style_emb_pt}
++train_config.align.loss_weight=0.05
++train_config.align.temperature=0.07
++train_config.align.use_global_negatives=true
++train_config.align.style_emb_on_gpu=true
""".strip()

argv = shlex.split(args)

# =========================
# 4) runpy 运行（参考你的模板）
# =========================
if argv[0] == "python":
    argv.pop(0)

# 这里只有脚本路径，不用 -m
fun = runpy.run_path

# 覆盖 sys.argv，让 hydra 正常解析
sys.argv = [argv[0]] + argv[1:]

# 可选：确保输出目录存在（日志/ckpt 写入时更稳）
os.makedirs(output_dir, exist_ok=True)

fun(argv[0], run_name="__main__")
