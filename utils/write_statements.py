from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import sys
import os
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
from torch.backends import cudnn
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils import *

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def add_template(questions, answers):
    messages = [
        {"role": "user", "content": "Give you a question and corresponding answer, turn it into a declarative sentence. Remember, don't answer the question, add any additional information or make mistakes in grammar!"},
        {"role": "assistant", "content": "Yes, I understand."},

        {"role": "user", "content": "Q: What gender is the person with a face? A: male."}, # TC
        {"role": "assistant", "content": "The person with a face is a male."},

        {"role": "user", "content": "Q: Is the person in white a sailboat in the river? A: no."}, # DO
        {"role": "assistant", "content": "The person in white is not a sailboat in the river."},

        {"role": "user", "content": "Q: Why do the cowboy clothes get down quickly? A: catch cow."}, # CW
        {"role": "assistant", "content": "The cowboy clothes get down quickly to catch cow."},

        {"role": "user", "content": "Q: Does the dancers dance indoors? A: yes."}, # DL
        {"role": "assistant", "content": "The dancers dance indoors."},

        {"role": "user", "content": "Q: how many people declare that the end is nigh? A: four."}, # TN
        {"role": "assistant", "content": "Four people declare that the end is nigh."},

        {"role": "user", "content": "Q: What is behind the person in the video? A: horse."}, # CH
        {"role": "assistant", "content": "Horse is behind the person in the video."},

        {"role": "user", "content": f"Q: {questions} A: {answers}."},
    ]
    return messages

init_seeds(42)

class MSVDQADataset(Dataset):
    def __init__(
        self,
        anno_path = '/home/whb/workspace/msvdQA/annotations/train_qa.json',
        mapper_path = "/home/whb/workspace/msvdQA/annotations/youtube_mapping.txt",
        video_path = "/home/whb/workspace/msvdQA/videos",
        frame_path = "/home/whb/workspace/msvdQA/frames_32",
        feature_path = "/home/whb/workspace/msvdQA/vision_features/feats_wo_norm.h5",
        frame_count = 32
    ):
        self.data = load_json('/home/whb/workspace/msvdQA/annotations/train_qa.json') + load_json('/home/whb/workspace/msvdQA/annotations/val_qa.json') + load_json('/home/whb/workspace/msvdQA/annotations/test_qa.json')
        self.mapper = {}
        f = open(mapper_path, 'r').read()
        lines = f.split('\n')
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    vid_id = parts[1]
                    video_name = parts[0]
                    self.mapper[vid_id] = video_name
        self.video_path = video_path
        self.frame_path = frame_path
        self.frame_count = frame_count

        self.video_ids = []
        self.videos = []
        self.frames = []
        self.questions = []
        self.answers_text = []
        self.answers = []
        self.types = []
        self.qids = []

        for data in self.data:
            
            temp_id = f"vid{data['video_id']}"
            video_id = self.mapper[temp_id]
            self.video_ids.append(video_id)
            self.qids.append(data['id'])
            self.types.append('N/A')
            self.questions.append(data['question'])
            self.answers.append(data['answer'])
            self.answers_text.append(data['answer'])

            self.videos.append(self.video_path + f"/{video_id}.mp4")
            self.frames.append(self.frame_path +f"/{video_id}")

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        type = str(self.types[index])
        question = str(self.questions[index])
        answer = str(self.answers[index])
        answer_text = str(self.answers_text[index])

    
        return {
                "video_ids": video_id,
                "qids": qid,
                "types": type,
                "questions": question,
                "answers": answer,
                "answers_text": answer_text,        
            }

class MSRVTTQADataset(Dataset):
    def __init__(
        self,
        anno_path = '/home/whb/workspace/msrvttQA/annotations/train_qa.json',
        video_path = "/home/whb/workspace/msrvttQA/videos",
        frame_path = "/home/whb/workspace/msrvttQA/frames_32",
        feature_path = "/home/whb/workspace/msrvttQA/vision_features/feats_wo_norm.h5",
        frame_count = 32
    ):
        self.data = load_json('/home/whb/workspace/msrvttQA/annotations/train_qa.json') + load_json('/home/whb/workspace/msrvttQA/annotations/val_qa.json') + load_json('/home/whb/workspace/msrvttQA/annotations/test_qa.json')
        self.video_path = video_path
        self.frame_path = frame_path
        self.frame_count = frame_count
        self.image_processor = image_transform(image_size=224)

        self.video_ids = []
        self.videos = []
        self.frames = []
        self.questions = []
        self.answers_text = []
        self.answers = []
        self.types = []
        self.qids = []

        for data in self.data:

            self.video_ids.append(data['video_id'])
            self.qids.append(data['id'])
            self.types.append(data['category_id'])
            self.questions.append(data['question'])
            self.answers.append(data['answer'])
            self.answers_text.append(data['answer'])

            self.videos.append(self.video_path + f"/video{data['video_id']}.mp4")
            self.frames.append(self.frame_path +f"/video{data['video_id']}")

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        type = str(self.types[index])
        question = str(self.questions[index])
        answer = str(self.answers[index])
        answer_text = str(self.answers_text[index])
    
        return {
                "video_ids": video_id,
                "qids": qid,
                "types": type,
                

                "questions": question,
                "answers": answer,
                "answers_text": answer_text,        
            }

class ActivityQADataset(Dataset):
    def __init__(
        self,
        anno_path = '/home/whb/workspace/activityQA/annotations/train.json',
        video_path = "/home/whb/workspace/activityQA/videos",
        frame_path = "/home/whb/workspace/activityQA/frames_32",
        frame_count = 32
    ):
        self.data = load_json('/home/whb/workspace/activityQA/annotations/train.json') + load_json('/home/whb/workspace/activityQA/annotations/val.json') + load_json('/home/whb/workspace/activityQA/annotations/test.json')
        self.video_path = video_path
        self.frame_path = frame_path
        self.frame_count = frame_count
        self.image_processor = image_transform(image_size=224)

        self.video_ids = []
        self.videos = []
        self.frames = []
        self.questions = []
        self.answers = []
        self.types = []
        self.qids = []

        for data in self.data:
            
            self.video_ids.append('v_'+data['video_name'])
            self.qids.append(data['question_id'])
            self.types.append(data['type'])
            self.questions.append(data['question'].capitalize()+"?")
            self.answers.append(data['answer'])

            self.videos.append(self.video_path + f"/v_{data['video_name']}.mp4")
            self.frames.append(self.frame_path +f"/v_{data['video_name']}")

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        type = str(self.types[index])
        question = str(self.questions[index])
        answer = str(self.answers[index])
        
        return {
                "video_ids": video_id,
                "qids": qid,
                "types": type,

                "questions": question,
                "answers": answer
            }
        
dataset = ActivityQADataset()   

device = "cuda:6" # the device to load the model onto
tokenizer = AutoTokenizer.from_pretrained("/home/whb/workspace/Mistral-7B-Instruct-v0.1/")
model = AutoModelForCausalLM.from_pretrained("/home/whb/workspace/Mistral-7B-Instruct-v0.1/", torch_dtype=torch.bfloat16)
model.to(device)

activityqa_statement = []
for entry in tqdm(dataset):
    video_id = entry['video_ids']
    qids = entry['qids']
    types = entry['types']
    questions = entry['questions']
    answers = entry['answers']

    encodeds = tokenizer.apply_chat_template(add_template(questions, answers), return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=False)
    decoded_sequence = tokenizer.decode(generated_ids[0][len(model_inputs[0]):], skip_special_tokens=True).replace('[/INST]', '')

    activityqa_statement.append({
        "video_ids": video_id,
        "qids": qids,
        "types": types,
        "questions": questions,
        "answers": answers,
        "statements": decoded_sequence,
    })

with open('activityqa_statement.json', 'w') as f:
    json.dump(activityqa_statement, f, indent=2)