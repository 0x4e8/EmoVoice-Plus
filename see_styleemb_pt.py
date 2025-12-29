import torch
import json
from pathlib import Path

# ====== 配置区 - 修改这里 ======
PT_FILE_PATH = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice/EmoVoice-DB/train.style_emb.pt"  # 替换为你的.pt文件实际路径
MAX_ITEMS_TO_SHOW = 10       # 要显示的最大条目数量
EMBEDDING_DIMS_TO_SHOW = 10 # 每个嵌入向量要显示的维度数量
# ============================

def inspect_pt_file():
    # 加载.pt文件
    print(f"加载文件: {PT_FILE_PATH}")
    data = torch.load(PT_FILE_PATH, map_location='cpu')
    
    # 1. 打印元数据
    print("\n" + "="*50)
    print("文件元数据")
    print("="*50)
    metadata_keys = ['version', 'model', 'audio_key', 'audio_root', 'granularity']
    for key in metadata_keys:
        if key in data:
            print(f"{key}: {data[key]}")
    
    # 2. 打印关键计数
    print("\n" + "="*50)
    print("数据统计")
    print("="*50)
    print(f"总键数量 (keys): {len(data.get('keys', []))}")
    print(f"有效条目 (items): {len(data.get('items', []))}")
    print(f"跳过条目 (skipped): {len(data.get('skipped', []))}")
    
    # 3. 显示嵌入张量信息
    if 'emb' in data:
        emb = data['emb']
        print("\n" + "="*50)
        print("嵌入张量详情")
        print("="*50)
        print(f"形状: {list(emb.shape)} (样本数={emb.shape[0]}, 特征维度={emb.shape[1]})")
        print(f"数据类型: {emb.dtype}")
        print(f"内存占用: {emb.element_size() * emb.nelement() / 1024**2:.2f} MB")
        
        # 显示部分嵌入值
        if emb.numel() > 0:
            print(f"\n前 {min(2, emb.shape[0])} 个样本的前 {EMBEDDING_DIMS_TO_SHOW} 个维度值:")
            for i in range(min(2, emb.shape[0])):
                values = emb[i, :EMBEDDING_DIMS_TO_SHOW].tolist()
                formatted = ", ".join(f"{v:.4f}" for v in values)
                print(f"  样本 {i} (键: {data['keys'][i]}): [{formatted} ...]")
    
    # 4. 显示条目示例
    if 'items' in data and data['items']:
        print("\n" + "="*50)
        print(f"有效条目示例 (显示前 {MAX_ITEMS_TO_SHOW} 个)")
        print("="*50)
        for i, item in enumerate(data['items'][:MAX_ITEMS_TO_SHOW]):
            print(f"条目 {i}:")
            print(json.dumps(item, indent=2, ensure_ascii=False))
    
    # 5. 显示跳过条目
    if 'skipped' in data and data['skipped']:
        print("\n" + "="*50)
        print(f"跳过条目示例 (显示前 {MAX_ITEMS_TO_SHOW} 个)")
        print("="*50)
        for i, skipped in enumerate(data['skipped'][:MAX_ITEMS_TO_SHOW]):
            print(f"跳过 {i}:")
            print(json.dumps(skipped, indent=2, ensure_ascii=False))
    
    # 6. 导出所有键到文本文件
    keys_file = Path(PT_FILE_PATH).with_suffix('.keys.txt')
    if 'keys' in data:
        with open(keys_file, 'w', encoding='utf-8') as f:
            for key in data['keys']:
                f.write(f"{key}\n")
        print(f"\n所有键已保存到: {keys_file.absolute()}")
        print(f"总键数量: {len(data['keys'])}")
    
    # 7. 重要提示
    print("\n" + "="*50)
    print("使用提示")
    print("="*50)
    print("1. 要查看完整键列表，请打开上面生成的 .keys.txt 文件")
    print("2. 要检查特定键的嵌入，可在Python中执行:")
    print(f"   data = torch.load('{PT_FILE_PATH}')")
    print("   target_key = 'your_key_here'")
    print("   idx = data['keys'].index(target_key)")
    print("   print(data['emb'][idx])")
    print("3. 要修改显示数量，编辑脚本顶部的配置变量")

if __name__ == "__main__":
    inspect_pt_file()