import random
from pathlib import Path

# ======================
# 配置参数 (直接修改这里)
# ======================
INPUT_FILE = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train.jsonl"
OUTPUT_FILE = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train_shuffled_5times.jsonl"
NUM_SHUFFLES = 5  # 连续打乱5次
SEED = 42  # 随机种子

def shuffle_dataset_n_times():
    """对数据集连续打乱N次，只保存最终结果"""
    # 1. 读取原始数据（保持行数不变）
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f if line.strip()]
    
    original_count = len(data)
    print(f"原始数据: {original_count} 行")
    
    # 2. 连续打乱5次（每次在上次结果基础上打乱）
    # random.seed(SEED)
    for i in range(NUM_SHUFFLES):
        random.seed(SEED + i)
        random.shuffle(data)  # 原地打乱
        print(f"  ✅ 完成第 {i+1} 次打乱 (种子={SEED + i})")
    
    # 3. 验证行数不变
    assert len(data) == original_count, "错误：行数发生变化！"
    print(f"✅ 最终行数验证: {len(data)} 行 (与原始文件一致)")
    
    # 4. 保存最终结果
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')
    
    print(f"\n✨ 成功! {NUM_SHUFFLES}次打乱后的数据已保存至:")
    print(f"   {output_path}")
    print(f"   (行数: {original_count}, 与原始文件完全相同)")

if __name__ == "__main__":
    shuffle_dataset_n_times()