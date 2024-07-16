import torch
from transformers import AutoConfig, StoppingCriteria
import random
import re
import os
import json
import requests
from PIL import Image
from io import BytesIO
import json
import pickle
import pandas as pd
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
from typing import Optional, Tuple, Any, Union, List

def _convert_to_rgb(image):
    return image.convert('RGB')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def get_frames(lst, M):

    frame_num = len(lst)

    if frame_num == 32:    
        if M==16:
            result = [lst[0], lst[2], lst[4], lst[6], lst[8], lst[10], lst[12], lst[14], lst[16], lst[18], lst[20], lst[22], lst[24], lst[26], lst[28], lst[30]]
        elif M==8:
            result = [lst[2], lst[6], lst[10], lst[14], lst[18], lst[22], lst[26], lst[30]]
        elif M==4:
            result = [lst[4], lst[12], lst[20], lst[28]]
        elif M==1:
            result = [lst[16]]
        else:
            result = lst

    elif frame_num == 16:
        if M==8:
            result = [lst[0], lst[2], lst[4], lst[6], lst[8], lst[10], lst[12], lst[15]]
        elif M==4:
            result = [lst[2], lst[6], lst[10], lst[14]]
        elif M==1:
            result = [lst[7]]
        else:
            result = lst        

    return result

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
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

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def compute_acc(bs, labels, preds):
    acc = 0
    for i in range(bs):
        label = labels[i]
        pred = preds[i]

        if pred.lower() == label.lower():
            acc += 1
    return acc/bs

def compute_acc_nextqa():
    folder_path = 'files/'
    target_prefix = 'nextqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    class_counts = {'C': 0, 'T': 0, 'D': 0}
    class_correct = {'C': 0, 'T': 0, 'D': 0}

    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
            class_correct[sample['type'][0]] += 1
        class_counts[sample['type'][0]] += 1 

    overall_accuracy = correct_predictions / total_samples
    class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}

    return overall_accuracy, class_accuracies

def compute_acc_intentqa():
    folder_path = 'files/'
    target_prefix = 'intentqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    class_counts = {'CW': 0, 'CH': 0, 'TN&TP': 0}
    class_correct = {'CW': 0, 'CH': 0, 'TN&TP': 0}

    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
            if sample['type'] == 'CW':
                class_correct['CW'] += 1
            elif sample['type'] == 'CH':
                class_correct['CH'] += 1
            elif 'T' in sample['type']:
                class_correct['TN&TP'] += 1
        if sample['type'] == 'CW':
            class_counts['CW'] += 1
        elif sample['type'] == 'CH':
            class_counts['CH'] += 1
        elif 'T' in sample['type']:
            class_counts['TN&TP'] += 1

    overall_accuracy = correct_predictions / total_samples
    class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}

    return overall_accuracy, class_accuracies

def compute_acc_starqa():
    folder_path = 'files/'
    target_prefix = 'starqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    class_counts = {'Int': 0, 'Fea': 0, 'Pre': 0, 'Seq': 0}
    class_correct = {'Int': 0, 'Fea': 0, 'Pre': 0, 'Seq': 0}

    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
            class_correct[sample['type']] += 1
        class_counts[sample['type']] += 1 

    class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}
    overall_accuracy = (class_accuracies['Int'] + class_accuracies['Fea'] + class_accuracies['Pre'] + class_accuracies['Seq']) / 4

    return overall_accuracy, class_accuracies

def compute_acc_trafficqa():
    folder_path = 'files/'
    target_prefix = 'trafficqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    class_counts = {'U': 0, 'A': 0, 'F': 0, 'R': 0, 'C': 0, 'I': 0}
    class_correct = {'U': 0, 'A': 0, 'F': 0, 'R': 0, 'C': 0, 'I': 0}

    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
            class_correct[sample['type']] += 1
        class_counts[sample['type']] += 1 

    class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy, class_accuracies

def compute_acc_vlep():
    folder_path = 'files/'
    target_prefix = 'vlep_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    class_counts = {'N/A.': 0}
    class_correct = {'N/A.': 0}

    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
            class_correct[sample['type']] += 1
        class_counts[sample['type']] += 1 

    class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy, class_accuracies

def compute_acc_msrvttqa():
    folder_path = 'files/'
    target_prefix = 'msrvttqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy

def compute_acc_msvdqa():
    folder_path = 'files/'
    target_prefix = 'msvdqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy

def compute_acc_activityqa():
    folder_path = 'files/'
    target_prefix = 'activityqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 
                
    total_samples = len(merged_data)
    correct_predictions = 0
    for sample in merged_data:
        if sample['pred'] == sample['label']:
            correct_predictions += 1
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy

def compute_acc_causalqa():
    folder_path = 'files/'
    target_prefix = 'causalqa_records_'
    # 初始化一个空的列表来存储合并后的数据
    merged_data = []
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(target_prefix) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data += data 

    merged_predictions = {}
    for sample in merged_data:
        video_id = sample['video_id']
        pred_type = sample['type']

        if video_id not in merged_predictions:
            merged_predictions[video_id] = {
                'video_id': video_id,
                'descriptive': {},
                'explanatory': {},
                'predictive_answer': {},
                'predictive_reason': {},
                'counterfactual_answer': {},
                'counterfactual_reason': {}
            }
        
        merged_predictions[video_id][pred_type]['input'] = sample['input']
        merged_predictions[video_id][pred_type]['label'] = sample['label']
        merged_predictions[video_id][pred_type]['pred'] = sample['pred']

    all_num = 0
    acc_descriptive = 0
    acc_explanatory = 0
    acc_predictive_answer = 0
    acc_predictive_reason = 0
    acc_counterfactual_answer = 0
    acc_counterfactual_reason = 0
    acc_predictive = 0
    acc_counterfactual = 0

    for key in merged_predictions.keys():
        if (merged_predictions[key]['descriptive'] != {}) and (merged_predictions[key]['explanatory'] != {}) and (merged_predictions[key]['predictive_answer'] != {}) and (merged_predictions[key]['predictive_reason'] != {}) and (merged_predictions[key]['counterfactual_answer'] != {}) and (merged_predictions[key]['counterfactual_reason'] != {}):
            all_num += 1
            predictive_answer = False
            predictive_reason = False
            counterfactual_answer = False
            counterfactual_reason = False
            if merged_predictions[key]['descriptive']['pred'] == merged_predictions[key]['descriptive']['label']:
                acc_descriptive += 1
            if merged_predictions[key]['explanatory']['pred'] == merged_predictions[key]['explanatory']['label']:
                acc_explanatory += 1
            if merged_predictions[key]['predictive_answer']['pred'] == merged_predictions[key]['predictive_answer']['label']:
                acc_predictive_answer += 1
                predictive_answer = True
            if merged_predictions[key]['predictive_reason']['pred'] == merged_predictions[key]['predictive_reason']['label']:
                acc_predictive_reason += 1
                predictive_reason = True
            if merged_predictions[key]['counterfactual_answer']['pred'] == merged_predictions[key]['counterfactual_answer']['label']:
                acc_counterfactual_answer += 1
                counterfactual_answer = True
            if merged_predictions[key]['counterfactual_reason']['pred'] == merged_predictions[key]['counterfactual_reason']['label']:
                acc_counterfactual_reason += 1
                counterfactual_reason = True
            if predictive_answer and predictive_reason:
                acc_predictive += 1
            if counterfactual_answer and counterfactual_reason:
                acc_counterfactual += 1

    class_accuracies = {
                        'D': acc_descriptive/all_num, 
                        'E': acc_explanatory/all_num,  
                        'PA': acc_predictive_answer/all_num, 
                        'PR': acc_predictive_reason/all_num,  
                        'PAR': acc_predictive/all_num,  
                        'CA': acc_counterfactual_answer/all_num,  
                        'CR': acc_counterfactual_reason/all_num,  
                        'CAR': acc_counterfactual/all_num
                        }
    overall_acc = (class_accuracies['D'] + class_accuracies['E'] + class_accuracies['PAR'] + class_accuracies['CAR'])/4
    return overall_acc, class_accuracies
