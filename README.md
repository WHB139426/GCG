# GCG - ACM MM'24
**Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering [ACM MM'24]**. This is the official implementation of the [[Paper](https://arxiv.org/abs/2401.10711)] accepted by ACM MM'24.

## Install

1. Clone this repository and navigate to GCG folder
```bash
git clone https://github.com/WHB139426/GCG.git
cd GCG
mkdir experiments
mkdir files
```

2. Install Package
```Shell
conda create -n gcg python=3.9.16
conda activate gcg
pip install -r requirements.txt
```

## Pretrained Weights of InstructBLIP

You can prepare the pretrained weights of InstructBLIP-T5-XL according to [[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)].

Since we have changed the structure of the code of the model, we recommend you download the pretrained weights of EVA-CLIP, and QFormer directly in [[🤗HF](https://huggingface.co/WHB139426/GCG/tree/main)]. The pretrained weights should be organized as follows,

```
├── GCG
│   └── experiments
│     └── eva_vit_g.pth
│     └── qformer_t5.pth
│     └── query_tokens_t5.pth
│     └── llm_proj_t5.pth
│     └── eva_vit_post_layernorm.pth
│     └── eva_clip_text_model.pth
│     └── eva_clip_last_vision_head.pth
│     └── eva_clip_last_vision_norm.pth
│     └── eva_clip_last_vision_block.pth
```

## Datasets
You should download the videos of NExT-QA from https://github.com/doc-doc/NExT-QA?tab=readme-ov-file or directly with the link [[videos](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view)]

We provide the annotation files in [[🤗HF](https://huggingface.co/WHB139426/GCG/tree/main)], and you should organize the data as follows,

```
├── nextqa
│   └── annotations_mc
│   └── frames_32
│   └── videos
│   └── vision_features
|   └── map_vid_vidorID.json
├── GCG
│   └── datasets
│   └── models
│   └──...
```
Then, you should extract 32 frames per video into the `nextqa/frames_32` folder with the python scripts

```Shell
python utils/extract_frames.py
```

After that, you should extract the video features in advance into the `nextqa/vision_features` with the python scripts

```Shell
python utils/extract_features.py
```

## Training

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1111 finetune_ans.py
```











