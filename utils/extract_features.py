from transformers import Blip2Config
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import  AutoTokenizer, Blip2Processor
import os
import sys
import numpy as np
import h5py
import cv2
import torch
from transformers import AutoConfig, StoppingCriteria
import random
import re
import requests
from PIL import Image
from io import BytesIO
import json
import pickle
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils import *
from models.eva_vit import Blip2VisionModel

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class NEXTQADataset(Dataset):
    def __init__(
        self,
        frame_path = "../nextqa/frames_32"
    ):
        self.frame_path = frame_path
        self.image_processor = image_transform(image_size=224)
        self.video_ids = os.listdir(self.frame_path)
        self.image_files = []
        self.image_ids = []
        for video_id in self.video_ids:
            frame_files = os.listdir(self.frame_path + f"/{video_id}")
            frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            for frame_file in frame_files:
                self.image_files.append(self.frame_path + f"/{video_id}/{frame_file}")
                self.image_ids.append(f"{video_id}_{frame_file.replace('.jpg','')}")

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_file = self.image_files[index]
        pixel_values = self.image_processor(Image.open(image_file))

        return {
                "image_ids": image_id,
                "image_files": image_file,
                "pixel_values": pixel_values,
            }


dataset=NEXTQADataset(frame_path = "../nextqa/frames_32")
print(len(dataset))
print(dataset[0]["image_ids"])
print(dataset[0]["image_files"])
print(dataset[0]["pixel_values"].shape)

blip2_config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
blip2_config.vision_config.torch_dtype = torch.float16
vision_model = Blip2VisionModel(blip2_config.vision_config)
vision_model.load_state_dict(torch.load("experiments/eva_vit_g.pth", map_location='cpu'))

data_loader = DataLoader(dataset=dataset, batch_size=768, shuffle=False, drop_last=False, num_workers=16)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vision_model.to(device)
vision_model.eval()

f = h5py.File('../nextqa/vision_features/feats_wo_norm_32.h5', "w")

for i, data in enumerate(tqdm(data_loader)):
    image_ids = data['image_ids']
    pixel_values = data['pixel_values'].to(device)
    # 抽取视觉特征
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        with torch.no_grad():
            frame_features = vision_model(pixel_values).last_hidden_state_without_norm # shape: [bs, 257, 1408]
    frame_features = frame_features.cpu().numpy()
    if i==0:
        print(frame_features.shape)
    for j in range(frame_features.shape[0]):
        f.create_dataset(f"{image_ids[j]}", data=frame_features[j])

