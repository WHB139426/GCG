import cv2
import os
from tqdm import tqdm
import sys
import json
import pandas as pd

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_csv(path):
    file_list = []
    data = pd.read_csv(path)
    columns = data.columns.tolist()
    for index, row in data.iterrows():
        file_list.append({})
        for column in columns:
            file_list[index][column] = row[column]
    return file_list

def extract_frames(video_path, frame_path, frames=16):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    # 获取视频帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 跳到指定的起始帧
    start_frame = 1
    # 计算抽帧间隔
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)-start_frame
    frame_interval = max(total_frames // frames, 1)
    # print(fps, start_frame, total_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT), frame_interval)

    if total_frames < frames:
        print(frame_path, f"<{frames} frames!!!!!!!")

    ret, frame = cap.read()
    # 开始抽帧
    frame_count = 0
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        # 每个frame_interval帧保存一帧
        if (current_frame-start_frame) % frame_interval == 0 and current_frame >= start_frame:
            output_file_path = os.path.join(frame_path, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_file_path, frame)
            frame_count += 1
        # 如果已经抽取足够帧数，提前结束循环
        if frame_count >= frames:
            break
    # 释放视频文件对象
    cap.release()

    if frame_count < frames:
        print(frame_path, f"<{frames} frames!!!!!!!")


train_data = load_csv("../nextqa/annotations_mc/train.csv")
val_data = load_csv("../nextqa/annotations_mc/val.csv")
test_data = load_csv("../nextqa/annotations_mc/test.csv")
mapper = load_json('../nextqa/map_vid_vidorID.json')
data = train_data + val_data + test_data

video_ids = []
for item in data:
    video_id = item['video']
    video_ids.append(video_id)
video_ids = list(set(video_ids))

for video_id in tqdm(video_ids):
    video_path = f"../nextqa/videos/{mapper[str(video_id)]}.mp4"
    frame_path = f"../nextqa/frames_32/{video_id}"
    extract_frames(video_path, frame_path, frames=32)
