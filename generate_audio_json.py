import os
import json

# ===== 硬编码参数配置区 =====
# 请根据实际需求修改以下参数
NEUTRAL_DIR = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/gene_trip_data/triplets_out/wav/neutral"
SAD_DIR = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/gene_trip_data/triplets_out/wav/sad"
OUTPUT_JSON = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/tools/data/audio_paths_sad.json"  # 输出的JSON文件名

# 要读取的文件数量 (m 和 n)
M = 1000000  # neutral目录下要读取的.wav文件数量
N = 1000000  # happy目录下要读取的.wav文件数量
# ===== 参数配置结束 =====

def get_wav_paths(directory, max_files):
    """
    获取目录下指定数量的.wav文件绝对路径
    
    Args:
        directory: 目标目录路径
        max_files: 要获取的文件数量
    
    Returns:
        按文件名排序的绝对路径列表（最多 max_files 个）
    """
    # 获取所有.wav文件（不区分大小写）
    wav_files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.wav')
    ]
    
    # 按文件名排序确保顺序一致
    wav_files.sort()
    
    # 限制文件数量，如果实际文件数不足则取全部
    selected_files = wav_files[:min(max_files, len(wav_files))]
    
    # 生成绝对路径
    return [os.path.abspath(os.path.join(directory, f)) for f in selected_files]

def main():
    # 获取指定数量的文件路径
    neutral_paths = get_wav_paths(NEUTRAL_DIR, M)
    happy_paths = get_wav_paths(SAD_DIR, N)
    
    # 构建JSON数据
    data = {
        "neutral": neutral_paths,
        "happy": happy_paths
    }
    
    # 写入JSON文件
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    
    # 打印统计信息
    print("=" * 50)
    print(f"成功生成JSON文件: {os.path.abspath(OUTPUT_JSON)}")
    print("-" * 50)
    print(f"neutral目录配置 (m={M}):")
    print(f"  目录路径: {NEUTRAL_DIR}")
    print(f"  请求文件数: {M}, 实际获取: {len(neutral_paths)}")
    for i, path in enumerate(neutral_paths, 1):
        print(f"    [{i}] {path}")
    
    print(f"\nhappy目录配置 (n={N}):")
    print(f"  目录路径: {SAD_DIR}")
    print(f"  请求文件数: {N}, 实际获取: {len(happy_paths)}")
    for i, path in enumerate(happy_paths, 1):
        print(f"    [{i}] {path}")
    print("=" * 50)

if __name__ == "__main__":
    main()