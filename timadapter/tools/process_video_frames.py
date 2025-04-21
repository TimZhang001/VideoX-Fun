import cv2
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

def get_random_frame(
    video_path: str, 
    target_size: Tuple[int, int] = (224, 224),
    max_retries: int = 3
) -> Optional[np.ndarray]:
    """
    从视频中随机抽取一帧并调整分辨率
    :param video_path: 视频绝对路径
    :param target_size: 目标分辨率 (width, height)
    :param max_retries: 随机帧读取失败时的最大重试次数
    :return: 调整后的帧 (numpy数组) 或 None（失败时）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return None

    # 获取视频总帧数（可能不准确）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 处理无效帧数的情况
    if total_frames <= 0:
        print(f"Warning: 检测到异常帧数 {total_frames}，尝试逐帧估算...")
        total_frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        if total_frames == 0:
            print(f"Error: 无法获取有效帧数 {video_path}")
            cap.release()
            return None
        cap.release()
        cap = cv2.VideoCapture(video_path)  # 重新打开视频

    for _ in range(max_retries):
        # 生成随机帧号（0-based）
        random_frame = random.randint(0, total_frames - 1)
        
        # 跳转到随机帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        
        if ret:
            # 调整分辨率
            resized_frame = cv2.resize(frame, target_size)
            cap.release()
            return resized_frame
    
    print(f"Error: 超过最大重试次数 {max_retries} 次，视频 {video_path}")
    cap.release()
    return None

def process_video_frames(
    train_video: List[str], 
    target_size: Tuple[int, int] = (224, 224)
) -> List[Optional[np.ndarray]]:
    """
    批量处理视频列表
    :param train_video: 视频路径列表
    :param target_size: 目标分辨率
    :return: 包含处理后的帧的列表（可能包含None）
    """
    processed_frames = []
    for path in train_video:
        frame = get_random_frame(path, target_size)
        processed_frames.append(frame)

    # numpy to PIL Image
    processed_frames = [Image.fromarray(frame) for frame in processed_frames]

    return processed_frames

# 使用示例 ------------------------------
if __name__ == "__main__":
    # 示例输入（替换为实际视频路径）
    train_video = [
        "/absolute/path/to/video1.mp4",
        "/absolute/path/to/video2.avi",
        "/invalid/path/demo.mov"  # 测试错误路径
    ]
    
    # 处理视频（输出可能包含None）
    frames = process_video_frames(train_video, target_size=(256, 256))
    
    # 统计结果
    success_count = sum(1 for f in frames if f is not None)
    print(f"成功处理 {success_count}/{len(train_video)} 个视频")
    
    # 查看第一个有效帧的形状（若存在）
    valid_frames = [f for f in frames if f is not None]
    if valid_frames:
        print("示例帧形状:", valid_frames[0].shape)