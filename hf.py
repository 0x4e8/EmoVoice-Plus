

# from huggingface_hub import snapshot_download
# import os

# snapshot_download(
#     repo_id="yhaha/EmoVoice-DB",
#     repo_type="dataset",              # 关键：这是数据集
#     local_dir="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice-DB",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     # allow_patterns=["evaluation/scan2cap/*.json"],  # 只下部分(可选)
#     # token="hf_xxx",                                     # 私有库时(可选)
# )
# print("done")

# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="yhaha/EmoVoice",
#     local_dir="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     allow_patterns=["EmoVoice_1.5B.pt"]   # 只下这个文件
# )


# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="yhaha/pretrain",
#     local_dir="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/ckpt",
#     local_dir_use_symlinks=False,
#     resume_download=True,
# )



from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="AIDC-AI/CSEMOTIONS",
    repo_type="dataset",              # 关键：这是数据集
    local_dir="/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/CSEMOTIONS",
    local_dir_use_symlinks=False,
    resume_download=True,
    # allow_patterns=["evaluation/scan2cap/*.json"],  # 只下部分(可选)
    # token="hf_xxx",                                     # 私有库时(可选)
)
print("done")