from torch.utils.data import Dataset
import random
import numpy as np
import torch
from transformers import  AutoTokenizer, InstructBlipProcessor
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import requests
from PIL import Image
from collections import Counter
from io import BytesIO
import json
import h5py
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
    
class NEXTQADataset(Dataset):
    def __init__(
        self,
        anno_path = '../nextqa/annotations_mc/train.csv',
        mapper_path = '../nextqa/map_vid_vidorID.json',
        video_path = "../nextqa/videos", 
        frame_path = "../nextqa/frames_32",
        feature_path = "../nextqa/vision_features/feats_wo_norm.h5",
        frame_count = 32
    ):
        
        self.data = load_csv(anno_path)
        self.mapper = load_json(mapper_path)
        self.video_path = video_path
        self.frame_path = frame_path
        self.frame_count = frame_count
        self.image_processor = image_transform(image_size=224)
        self.image_features = h5py.File(feature_path, "r")

        self.video_ids = []
        self.videos = []
        self.frames = []
        self.questions = []
        self.answers_option = []
        self.answers_text = []
        self.answers_ids = []
        self.types = []
        self.qids = []
        self.options_a0 = []
        self.options_a1 = []
        self.options_a2 = []
        self.options_a3 = []
        self.options_a4 = []

        for data in self.data:

            self.video_ids.append(data['video'])
            self.qids.append(data['qid'])
            self.types.append(data['type'])
            self.questions.append(data['question']+"?")
            self.options_a0.append(data['a0'])
            self.options_a1.append(data['a1'])
            self.options_a2.append(data['a2'])
            self.options_a3.append(data['a3'])
            self.options_a4.append(data['a4'])

            self.answers_ids.append(data['answer'])
            self.answers_text.append(data[f"a{str(data['answer'])}"] )
            self.answers_option.append(["A", "B", "C", "D", "E"][data['answer']])
            self.videos.append(self.video_path + f"/{self.mapper[str(data['video'])]}.mp4")
            self.frames.append(self.frame_path +f"/{str(data['video'])}")

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        type = str(self.types[index])
        question = str(self.questions[index])
        option_a0 = str(self.options_a0[index])
        option_a1 = str(self.options_a1[index])
        option_a2 = str(self.options_a2[index])
        option_a3 = str(self.options_a3[index])
        option_a4 = str(self.options_a4[index])
        answer_id = self.answers_ids[index]
        answer_text = str(self.answers_text[index])
        answer_option = str(self.answers_option[index])

        frame_files = os.listdir(str(self.frames[index]))
        frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        frame_files = get_frames(frame_files, self.frame_count)

        frame_features = []
        for i in range(len(frame_files)):
            frame_features.append(torch.from_numpy(self.image_features[f"{video_id}_{frame_files[i].replace('.jpg','')}"][:]))
        frame_features = torch.stack(frame_features, dim=0) # [frame_count, 257, 1408]
        
        return {
                "video_ids": video_id,
                "qids": qid,
                "types": type,

                "frame_features": frame_features,

                "questions": question,
                "options_a0": option_a0,
                "options_a1": option_a1,
                "options_a2": option_a2,
                "options_a3": option_a3,
                "options_a4": option_a4,
                "answers_id": answer_id,
                "answers_text": answer_text,
                "answers": answer_option,

            }



