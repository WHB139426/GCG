import os
import shutil
import random
import torch
from tqdm import tqdm

def check_frames(source_dir, tgt_frame_num):
    all_num = 0
    count = 0
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        jpg_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")]
        if len(jpg_files) < tgt_frame_num:
            count += 1
            print(subdir_path)
        all_num += len(jpg_files)
    return count, all_num

def generate_uniform_elements(T, W):
    return torch.linspace(0, T-1, W, dtype=torch.int)

def move_files(source_dir, target_dir, tgt_frame_num):
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    # 遍历源文件夹下的每个子文件夹
    for subdir in tqdm(os.listdir(source_dir)):
        subdir_path = os.path.join(source_dir, subdir)
        target_subdir_path = os.path.join(target_dir, subdir)
        if not os.path.exists(target_subdir_path):
            os.makedirs(target_subdir_path, exist_ok=True)
            # 列出当前子文件夹下的jpg文件
            jpg_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")]
            jpg_files = sorted(jpg_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            # 如果jpg文件数量超过tgt_frame_num，则均匀取tgt_frame_num个文件；否则重复旧文件填充至tgt_frame_num个文件
            selected_files = []
            if len(jpg_files) >= tgt_frame_num:
                # 均匀选择tgt_frame_num个文件
                selected_indices = generate_uniform_elements(len(jpg_files), tgt_frame_num)
                selected_files = [jpg_files[i] for i in selected_indices]
                for filename in selected_files:
                    src_path = os.path.join(subdir_path, filename)
                    dest_path = os.path.join(target_subdir_path, filename)
                    shutil.copy(src_path, dest_path)
            else:
                # 重复旧文件填充至tgt_frame_num个文件
                max_value = len(jpg_files)-1
                selected_files = jpg_files
                for filename in selected_files:
                    src_path = os.path.join(subdir_path, filename)
                    dest_path = os.path.join(target_subdir_path, filename)
                    shutil.copy(src_path, dest_path)
                for i in range(tgt_frame_num - len(jpg_files)):
                    src_path = os.path.join(subdir_path, random.choice(jpg_files))
                    dest_path = os.path.join(target_subdir_path, f"frame_{max_value+i+1}.jpg")
                    shutil.copy(src_path, dest_path)
    print("转移完成。")


