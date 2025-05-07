import cv2
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import os

def get_random_frame(
    video_path: str, 
    target_size: Tuple[int, int] = (224, 224), # 高度 宽度 
    frame_idx : int = -1,
    adaptive_size: bool = False,
) -> Optional[np.ndarray]:
    """
    从视频中随机抽取一帧并调整分辨率
    :param video_path: 视频绝对路径
    :param target_size: 目标分辨率 (height， width)
    :return: 调整后的帧 (numpy数组) 或 None（失败时）
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error: 无法打开视频 {video_path}"

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames > 0, f"Error: 无效的帧数 {total_frames} 从视频 {video_path}"

    # 生成随机帧号（0-based）
    if frame_idx == -1 or frame_idx >= total_frames:
        random_frame = random.randint(0, total_frames - 1)
    else:
        random_frame = frame_idx
    
    # 跳转到随机帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = cap.read()
    
    if ret:
        # 自适应方向调整（修正逻辑错误）OpenCV 图像的 shape 属性始终按 ​​(高度, 宽度, 通道数)​​ 排列
        img_height,img_width = frame.shape[:2]
        
        # 判断原始图和目标尺寸方向是否一致（宽高比方向）
        if adaptive_size and (img_width > img_height) != (target_size[1] > target_size[0]):
            target_size = (target_size[1], target_size[0])  # 方向不一致则翻转尺寸

        # 保持比例缩放并填充黑色（修复尺寸判断逻辑）
        target_h, target_w = target_size[0], target_size[1]  # 高度 宽度
        
        # 只有当前尺寸与目标尺寸不一致时才调整（正确比较宽高）
        if (img_width, img_height) != (target_w, target_h):
            scale = min(target_w / img_width, target_h / img_height)
            new_w = int(img_width  * scale)
            new_h = int(img_height * scale)
            
            # 缩放图像
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建填充后的图像
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            dx, dy = (target_w - new_w) // 2, (target_h - new_h) // 2
            padded[dy:dy+new_h, dx:dx+new_w] = resized
            frame  = padded

        # 转换为RGB格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转化为PIL Image
        frame = Image.fromarray(frame)
        cap.release()
        return frame
    else:
        print(f"Error: 无法读取帧 {random_frame} 从视频 {video_path}")
        cap.release()
        return None
    
def process_video_frames(
    train_video_path: str, 
    target_size: Tuple[int, int] = (224, 224), # 高度 宽度 
    adpative_size: bool = False,
) -> List[Optional[np.ndarray]]:
    """
    批量处理视频列表
    :param train_video: 视频路径列表
    :param target_size: 目标分辨率
    :return: 包含处理后的帧的列表（可能包含None）
    """
    processed_frames = get_random_frame(train_video_path, target_size, -1, adpative_size)
    if processed_frames is None:
        print(f"Error: 无法处理视频 {train_video_path}")

    return processed_frames


def process_video_frames_2_dir(
    train_video: List[str], 
    target_size: Tuple[int, int] = (224, 224), # 高度 宽度
    adaptive_size: bool = False,
) -> List[Optional[np.ndarray]]:
    """
    批量处理视频列表
    :param train_video: 视频路径列表
    :param target_size: 目标分辨率
    :return: 包含处理后的帧的列表（可能包含None）
    """
    
    #assert len(train_video) == 1, "Only one video is allowed. len = {}".format(train_video)
    
    print("train_video is: " + train_video + "\n")
    if train_video.endswith(".mp4"):
        processed_frames = get_random_frame(train_video, target_size, 0, adaptive_size)
    elif train_video.lower().endswith((".jpg", ".jpeg", ".png")):
        processed_frames = Image.open(train_video)
        original_width, original_height = processed_frames.size # 宽度 高度
        
        # 自适应尺寸方向调整
        if adaptive_size and (original_width > original_height) != (target_size[1] > target_size[0]):
            target_size = (target_size[1], target_size[0])  # 翻转尺寸
        
        # 保持比例缩放并填充黑色 --- 新增部分 ---
        if (original_width, original_height) != (target_size[1], target_size[0]):
            # 计算缩放比例
            width_ratio  = target_size[1] / original_width
            height_ratio = target_size[0] / original_height
            scale        = min(width_ratio, height_ratio)
            
            # 计算新尺寸
            new_width  = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 缩放图像
            resized = processed_frames.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            
            # 创建填充后的画布
            padded = Image.new('RGB', (target_size[1], target_size[0]), color='black')
            
            # 计算粘贴位置
            paste_x = (target_size[1] - new_width) // 2
            paste_y = (target_size[0] - new_height) // 2
            
            # 粘贴缩放后的图像
            padded.paste(resized, (paste_x, paste_y))
            processed_frames = padded

    else:
        assert False, "Invalid video format. Only .mp4 and .jpg are supported."

    # 保存到文件夹 当前文件夹下的临时目录tmp/output_frames中
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "output_frames")
    os.makedirs(output_dir, exist_ok=True)
    abs_file_path = os.path.join(output_dir, "output_frames_{}.jpg".format(random.randint(1, 1000000)))
    processed_frames.save(abs_file_path)
    dst_size = [processed_frames.size[1], processed_frames.size[0]] # 高度 宽度
    return abs_file_path, dst_size

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