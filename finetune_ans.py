import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from torch import cuda
import time
from utils import *
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import pickle
import random
import json
from datasets.nextqa import NEXTQADataset
from torch.backends import cudnn
from utils.utils import *
from utils.optims import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='experiments')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    parser.add_argument('--word_size', default=4, help="n_gpus")
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--grounding_lr', type=float, default=7e-5) # 3e-5

    parser.add_argument('--use_schedule', type=bool, default=False)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-8)
    parser.add_argument('--min_lr', type=float, default=5e-6, help='min_lr for consine annealing')
    parser.add_argument('--max_T', type=int, default=30, help='epoches for lr->min_lr / min_lr->lr')
    parser.add_argument('--eval_step', type=int, default=1, help="eval every 1/eval_step epoch")
    parser.add_argument('--save_ckpt', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='nextqa', choices=['nextqa'])
    parser.add_argument('--frame_count', type=int, default=32)
    parser.add_argument('--mode', type=str, default='grounding', choices=['grounding', 'uniform', 'oracle'])

    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--width', type=float, default=0.2)
    parser.add_argument('--use_spatial', type=bool, default=True)
    parser.add_argument('--model', type=str, default='t5-xl', choices=['t5-xl'])
    parser.add_argument('--use_vit', type=bool, default=False)
    parser.add_argument('--use_lora', type=bool, default=False)

    args = parser.parse_args()
    return args
        
def reduce_metric(metric):
    metric_tensor = torch.tensor(metric).cuda(args.local_rank)
    dist.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
    metric = metric_tensor.item() / dist.get_world_size()
    return metric

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

def prepare_inputs(args, data):

    video_ids = data["video_ids"]
    qids = data["qids"]
    types = data["types"]

    questions = data["questions"] 
    answers = data["answers"]

    mc_prompt = "Considering the information presented in the frame, select the correct answer from the options."

    if args.dataset == 'intentqa':
        options_a0 = data["options_a0"]
        options_a1 = data["options_a1"]
        options_a2 = data["options_a2"]
        options_a3 = data["options_a3"]
        options_a4 = data["options_a4"]
        text_input =  ['Question: ' + question + f'\nOptions: \nA: {option_a0} \nB: {option_a1} \nC: {option_a2} \nD: {option_a3} \nE: {option_a4}' + '\nAnswer: ' for question, option_a0, option_a1, option_a2, option_a3, option_a4 in zip(questions, options_a0, options_a1, options_a2, options_a3, options_a4)]

    text_output = answers

    if args.dataset == 'nextqa':
        return text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4

@torch.no_grad()
def eval(args, val_loader, model):
    model.eval()

    val_loss = 0
    val_vqa_loss = 0
    val_reg_loss = 0
    val_info_loss = 0
    val_acc = 0
    overall_acc = 0

    acc_records = []

    for step, data in enumerate(val_loader):

        if args.dataset == 'nextqa':
            text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4 = prepare_inputs(args, data)
            samples = {
                    "text_input": text_input,
                    "text_output": text_output,
                    "questions": questions,
                    "options_a0": options_a0,
                    "options_a1": options_a1,
                    "options_a2": options_a2,
                    "options_a3": options_a3,
                    "options_a4": options_a4,
                    "frame_features": data["frame_features"].cuda(args.local_rank),
                    "answers_text": data["answers_text"],
                    "answers_id": data["answers_id"]
                }

        generate_kwargs = {
            "do_sample": True,
            "num_beams": 5, 
            "min_length": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 30,
            "temperature":1,
            "top_p":0.9,
            "repetition_penalty":1,
            "length_penalty":1
            }

        with torch.cuda.amp.autocast(enabled=True, dtype=model.module.dtype): # 前后开启autocast
            with torch.no_grad():
                outputs = model(samples)
                pred_texts = model.module.generate(samples, **generate_kwargs)

        for i in range(args.eval_bs):
            qid = data['qids'][i]
            video_id = data['video_ids'][i]
            type = data['types'][i]
            input_text = text_input[i]
            label = text_output[i]
            pred = pred_texts[i]

            acc_records.append({
                'qid': qid,
                'video_id': video_id,
                'type': type,
                'input': input_text,
                'label': label,
                'pred': pred
                })

        loss = outputs['loss']
        val_loss += loss.item() 
        val_vqa_loss += outputs['vqa_loss'].item() 
        val_reg_loss += outputs['regression_loss'].item() 
        val_info_loss += outputs['infoNCE_loss'].item() 
        val_acc += compute_acc(bs = args.eval_bs, labels = text_output, preds = pred_texts)

        if dist.get_rank() == 0 and step<=4:
            for i in range(len(text_input)):
                print()
                print("---------------------eval-------------------------")
                print("---------------------ids-------------------------")
                print("video_id: " + data["video_ids"][i] + "  qid: " + data["qids"][i])
                print("---------------------type-------------------------")
                print(data["types"][i])
                print("---------------------input-------------------------")
                print(text_input[i])
                print("---------------------preds-------------------------")
                print(pred_texts[i])
                print("--------------------answers------------------------")
                print(text_output[i])
                print()

    with open(f'files/{args.dataset}_records_{dist.get_rank()}.json', 'w') as f:
        json.dump(acc_records, f, indent=2)

    # 同步所有进程
    dist.barrier()
    
    for r in range(dist.get_world_size()):
        if dist.get_rank() == r:
            if len(os.listdir('files/')) >= dist.get_world_size():
                if args.dataset == 'nextqa':
                    overall_acc, class_acc = compute_acc_nextqa()
                    if dist.get_rank() == 0:
                        print('Overall Acc: ', overall_acc)
                        print('Class Acc: ', class_acc)
 
    # 同步所有进程
    dist.barrier()
    if dist.get_rank() == 0:
        folder_path = 'files/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)  

    # 对不同进程上的评价指标进行平均
    val_loss = round(reduce_metric(val_loss)/len(val_loader), 4)
    val_vqa_loss = round(reduce_metric(val_vqa_loss)/len(val_loader), 4)
    val_reg_loss = round(reduce_metric(val_reg_loss)/len(val_loader), 4)
    val_info_loss = round(reduce_metric(val_info_loss)/len(val_loader), 4)
    val_acc = round(reduce_metric(val_acc)/len(val_loader), 4)
    model.train()
    return val_loss, val_vqa_loss, val_reg_loss, val_info_loss, val_acc, overall_acc


def train(args, train_dataset, val_dataset, model):

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, pin_memory=True, shuffle=False, drop_last=True, num_workers=4)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_bs, sampler=val_sampler, pin_memory=True, shuffle=False, drop_last=True, num_workers=4)


    if args.mode == 'grounding':
        ignored_params = list(map(id, model.module.grounding.parameters())) # 返回的是parameters的 内存地址
        base_params = filter(lambda p: p.requires_grad and id(p) not in ignored_params, model.parameters()) 
        optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': model.module.grounding.parameters(), 'lr': args.grounding_lr}], 
        lr = args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    lr_schedule = LinearWarmupCosineLRScheduler(optimizer, max_epoch=args.max_T, min_lr=args.min_lr, init_lr=args.lr, warmup_steps=int(len(train_loader)/4), warmup_start_lr=args.warmup_start_lr)

    max_acc = 0

    scaler = torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象

    for epoch in range(args.epoch):

        model.train()
        # 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        train_loader.sampler.set_epoch(epoch)
        start = time.time()
        train_loss = 0
        train_vqa_loss = 0
        train_reg_loss = 0
        train_info_loss = 0
        train_acc = 0
        for step, data in enumerate(tqdm(train_loader, disable=not dist.get_rank() == 0)):
            model.train()
            if args.dataset == 'nextqa':
                text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4 = prepare_inputs(args, data)
                samples = {
                        "text_input": text_input,
                        "text_output": text_output,
                        "questions": questions,
                        "options_a0": options_a0,
                        "options_a1": options_a1,
                        "options_a2": options_a2,
                        "options_a3": options_a3,
                        "options_a4": options_a4,
                        "frame_features": data["frame_features"].cuda(args.local_rank),
                        "answers_text": data["answers_text"],
                        "answers_id": data["answers_id"]
                    }

            with torch.cuda.amp.autocast(enabled=True, dtype=model.module.dtype): # 前后开启autocast
                outputs = model(samples)
                with torch.no_grad():
                    # pred_texts = model.module.generate(samples, **generate_kwargs)
                    pred_texts = ['N/A' for i in range(args.bs)]

            loss = outputs['loss']
            train_loss += outputs['loss'].item() 
            train_vqa_loss += outputs['vqa_loss'].item() 
            train_reg_loss += outputs['regression_loss'].item() 
            train_info_loss += outputs['infoNCE_loss'].item() 
            train_acc += compute_acc(bs = args.bs, labels = text_output, preds = pred_texts)

            scaler.scale(loss).backward()  #为了梯度放大
            scaler.step(optimizer)
            scaler.update()  #准备着，看是否要增大scaler

            if args.use_schedule:
                lr_schedule.step(cur_epoch=epoch, cur_step=step)

            optimizer.zero_grad()                           

            if step % int(len(train_loader)/args.eval_step) == 0 and epoch > 0 and step >= int(len(train_loader)/args.eval_step) and step < len(train_loader)*0.9:
                val_loss, val_acc, overall_acc = eval(args, val_loader, model)
                if dist.get_rank() == 0:
                    print('epoch:{}/{} step:{}  val_loss:{} val_acc:{}'
                        .format(epoch + 1, args.epoch, step, val_loss, val_acc))
                if (overall_acc >= max_acc):    
                    max_acc = overall_acc
                    if args.save_ckpt:
                        torch.save(model.module.state_dict(), './{}/{}_{}_{}.pth'.format(args.experiment_path, f'{args.model}_{args.dataset}', epoch+1, overall_acc))

        # 对不同进程上的评价指标进行平均
        train_loss = round(reduce_metric(train_loss)/len(train_loader), 4)
        train_vqa_loss = round(reduce_metric(train_vqa_loss)/len(train_loader), 4)
        train_reg_loss = round(reduce_metric(train_reg_loss)/len(train_loader), 4)
        train_info_loss = round(reduce_metric(train_info_loss)/len(train_loader), 4)
        train_acc = round(reduce_metric(train_acc)/len(train_loader), 4)
        val_loss, val_vqa_loss, val_reg_loss, val_info_loss, val_acc, overall_acc = eval(args, val_loader, model)
        
        end = time.time()
        if dist.get_rank() == 0:
            print('epoch:{}/{}  time:{}h  lr:{}  batchsize:{}  train_loss:{}  val_loss:{}  train_acc: {}  val_acc:{}'
                .format(epoch + 1, args.epoch, str(round((end-start)/3600, 2)), args.lr, args.bs, train_loss, val_loss, train_acc, val_acc))
            print('train_vqa_loss:{}  train_reg_loss:{}  train_info_loss: {}  val_vqa_loss:{}  val_reg_loss:{}  val_info_loss: {}'
                .format(train_vqa_loss, train_reg_loss, train_info_loss, val_vqa_loss, val_reg_loss, val_info_loss))
            if (overall_acc >= max_acc):    
                max_acc = overall_acc
                if args.save_ckpt:
                    torch.save(model.module.state_dict(), './{}/{}_{}_{}.pth'.format(args.experiment_path, f'{args.model}_{args.dataset}', epoch+1, overall_acc))

    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'nextqa':
        train_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/train.csv', frame_count=args.frame_count)
        val_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/val.csv', frame_count=args.frame_count)
        test_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/test.csv', frame_count=args.frame_count)


    from models.blip2_t5_instruct import Blip2T5Instruct
   
    if 't5' in args.model:
        model = Blip2T5Instruct(
            dtype=torch.bfloat16,
            frame_num=args.frame_count,
            mode = args.mode,
            window_size = args.window_size,
            use_spatial = args.use_spatial,
            model = args.model,
            temperature = args.temperature,
            width = args.width,
            use_vit = args.use_vit,
            use_lora = args.use_lora
        )        
       

    device = torch.device('cuda', args.local_rank)
    dist.init_process_group(backend='nccl',rank=args.local_rank, world_size=args.word_size)
    init_seeds(args.seed + torch.distributed.get_rank())
    torch.cuda.set_device(device)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank, 
                                                        )

    if dist.get_rank() == 0:
        print(get_parameter_number(model))
        print("trian_num: ", len(train_dataset), " val_num: ", len(val_dataset),  " test_num: ", len(test_dataset))
        print(args)

    train(args, train_dataset, test_dataset, model)