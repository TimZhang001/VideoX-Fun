import os
import subprocess
from argparse import ArgumentParser

def find_common_mp4_files(input_dirs):
    """精确匹配各文件夹中的同名MP4，按输入顺序返回路径列表"""
    # 收集各文件夹中的MP4文件集合
    dir_files = []
    for dir_path in input_dirs:
        files = set()
        for f in os.listdir(dir_path):
            if f.lower().endswith(".mp4"):
                files.add(f)
        dir_files.append(files)
    
    # 计算所有文件夹共有的文件名（交集）
    common_files = set(dir_files[0])
    for files in dir_files[1:]:
        common_files &= files
    
    # 按输入文件夹顺序构建路径列表
    result = {}
    for filename in common_files:
        ordered_paths = []
        for dir_path in input_dirs:
            full_path = os.path.join(dir_path, filename)
            ordered_paths.append(full_path)
        result[filename] = ordered_paths
    return result

def hstack_videos(input_paths, output_path):
    """按固定顺序拼接视频"""
    # 生成输入文件列表
    input_args = []
    for path in input_paths:
        input_args.extend(["-i", path])
    
    # 构建按顺序拼接的滤镜链
    filter_complex = f"hstack=inputs={len(input_paths)}"
    
    cmd = [
        "ffmpeg",
        "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    return result.returncode == 0

def main():
    parser = ArgumentParser(description="按输入顺序拼接视频")
    parser.add_argument("-i", "--input_dirs", nargs="+", required=True,
                        help="输入文件夹顺序（第一个在最左边）")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="输出目录")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    common_files = find_common_mp4_files(args.input_dirs)
    if not common_files:
        print("错误：输入文件夹中没有共同MP4文件")
        return
    
    for filename, paths in common_files.items():
        output_path = os.path.join(args.output_dir, filename)
        print(f"正在拼接: {filename}")
        print("输入顺序：")
        for i, p in enumerate(paths, 1):
            print(f"  [{i}] {p}")
        
        if hstack_videos(paths, output_path):
            print(f"成功保存到：{output_path}\n")
        else:
            print(f"拼接失败：{filename}\n")

if __name__ == "__main__":
    main()