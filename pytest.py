#!/usr/bin/env python3
import os
import sys

def print_directory_tree(root_path, prefix="", is_last=True):
    """递归打印美观的目录树"""
    if not os.path.exists(root_path):
        print(f"{prefix}{'└── ' if is_last else '├── '}[!] 目录不存在: {os.path.basename(root_path)}")
        return 0
    
    items = sorted([item for item in os.listdir(root_path) 
                   if not item.startswith('.')])  # 跳过隐藏文件
    
    file_count = 0
    dir_count = 0
    
    for i, item in enumerate(items):
        path = os.path.join(root_path, item)
        is_current_last = (i == len(items) - 1)
        
        # 确定连接符
        connector = "└── " if is_current_last else "├── "
        child_prefix = prefix + ("    " if is_current_last else "│   ")
        
        # 打印条目
        print(f"{prefix}{connector}{item}")
        
        # 递归处理子目录
        if os.path.isdir(path):
            dir_count += 1
            sub_files, sub_dirs = print_directory_tree(path, child_prefix, is_current_last)
            file_count += sub_files
            dir_count += sub_dirs
        else:
            file_count += 1
    
    return file_count, dir_count

if __name__ == "__main__":
    base_dir = "/data/home/zdhs0047/zdhs0047_src_data/zengfanshuo/EmoVoice"
    target_dirs = ["examples", "src"]
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"[错误] 基础目录不存在: {base_dir}")
        sys.exit(1)
    
    total_files = 0
    total_dirs = 0
    
    print(f"📁 文件路径树 (仅限 {', '.join(target_dirs)}):")
    print("="*60)
    
    for subdir in target_dirs:
        full_path = os.path.join(base_dir, subdir)
        print(f"\n{subdir}/")
        
        if not os.path.exists(full_path):
            print(f"  └── [!] 目录不存在")
            continue
        
        # 打印子目录树
        files, dirs = print_directory_tree(full_path, "  ")
        total_files += files
        total_dirs += dirs
    
    print("\n" + "="*60)
    print(f"📊 总计: {total_files} 个文件, {total_dirs} 个子目录 (在 {', '.join(target_dirs)} 中)")