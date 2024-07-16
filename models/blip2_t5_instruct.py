import sys
import os
import contextlib
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast as autocast
from transformers import T5Tokenizer, BertTokenizer
from transformers import T5Config, Blip2Config, BertConfig
from transformers.modeling_outputs import BaseModelOutput
from einops import rearrange, repeat
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
import torch.nn.functional as F
from models.modeling_t5 import T5ForConditionalGeneration
from models.Qformer import BertLMHeadModel
from models.eva_vit import Blip2VisionModel
from models.eva_clip_branch_encoder import Clip_Branch_Encoder
from utils.simple_tokenizer import tokenize as clip_text_tokenizer

class Blip2T5Instruct(nn.Module):
    def __init__(
        self,
        frame_num = 32,
        dtype=torch.bfloat16,
        mode = 'grounding',
        window_size = 4,
        use_spatial = False,
        model = 't5-xl',
        temperature = 0.1,
        use_vit = False,
        width = 0.2,
        use_lora=False
    ):
        super().__init__()

        self.dtype = dtype
        self.frame_num = frame_num
        self.mode = mode
        self.window_size = window_size
        self.use_spatial = use_spatial
        self.model = model
        self.max_input_txt_len = 512
        self.max_output_txt_len = 512
        self.width = width
        self.temperature = temperature
        self.use_vit = use_vit
        self.use_lora = use_lora

        print('loading ViT')
        blip2_config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
        self.eva_vit_post_layer_norm = nn.LayerNorm(blip2_config.vision_config.hidden_size, eps=blip2_config.vision_config.layer_norm_eps)
        self.eva_vit_post_layer_norm.load_state_dict(torch.load("experiments/eva_vit_post_layernorm.pth", map_location='cpu'))
        if self.use_vit:
            blip2_config.vision_config.torch_dtype = self.dtype
            self.vision_model = Blip2VisionModel(blip2_config.vision_config)            
            self.vision_model.load_state_dict(torch.load("experiments/eva_vit_g.pth", map_location='cpu'))
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False
            self.vision_model.eval()

        print('loading Qformer')
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.Qformer = self.init_Qformer(num_query_token=32, vision_width=blip2_config.vision_config.hidden_size, cross_attention_freq=2)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        if self.model == 't5-xl':
            self.Qformer.load_state_dict(torch.load("experiments/qformer_t5.pth", map_location='cpu'))
            self.query_tokens = nn.Parameter(torch.load("experiments/query_tokens_t5.pth", map_location='cpu'))

        if self.mode == 'grounding' or self.mode == 'oracle':
            print('loading eva clip branch encoder')
            self.branch_encoder = Clip_Branch_Encoder()
            for name, param in self.branch_encoder.named_parameters():
                param.requires_grad = False
            self.branch_encoder.eval()

            from models.Transformer import create_sinusoidal_embeddings
            print('loading frame_embeds')        
            self.frame_embeds = nn.Embedding(self.frame_num, 768)
            create_sinusoidal_embeddings(
                n_pos=self.frame_num,
                dim=768,
                out=self.frame_embeds.weight,
            )

        if self.mode == 'grounding':
            print('loading Grounding')        
            from models.grounding_module import Grounding
            self.grounding = Grounding(dim=1024, heads=4, dropout=0.3, window_size=self.window_size, frame_num=self.frame_num, width=self.width, temperature=self.temperature)

        print('loading T5')
        if self.model == 't5-xl':
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", truncation_side='left')
            self.t5_output_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", truncation_side='right')
            t5_config = T5Config.from_pretrained("google/flan-t5-xl")
            t5_config.dense_act_fn = "gelu"
            self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", config=t5_config, torch_dtype=self.dtype)

        print('loading llm_proj')
        self.t5_proj = nn.Linear(768, self.t5_model.config.hidden_size)
        if self.model == 't5-xl':
            self.t5_proj.load_state_dict(torch.load("experiments/llm_proj_t5.pth", map_location='cpu'))

        print("Frozen ViT")
        for name, param in self.eva_vit_post_layer_norm.named_parameters():
            param.requires_grad = False
        self.eva_vit_post_layer_norm.eval()

        if self.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            print("LORA LLM")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False if self.training else True, r=16, lora_alpha=32, lora_dropout=0.05, 
                target_modules = ["q", "v"],
            )
            self.t5_model = get_peft_model(self.t5_model, peft_config)
        else:
            print("Frozen t5")
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16() 
            self.t5_model.eval()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def init_tokenizer(self, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.torch_dtype = self.dtype
        Qformer = BertLMHeadModel(config=encoder_config)
        return Qformer

    def HardTopK(self, k, x):
        topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, k
        indices = torch.sort(indices, dim=-1).values
        return indices    

    def generate_gauss_weight(self, center, width):
        # code copied from https://github.com/minghangz/cpl
        weight = torch.linspace(0, 1, self.frame_num)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9       
        w = 0.3989422804014327 #1/(math.sqrt(2*math.pi))
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        return weight
    
    def return_gmm_scores(self, label_probs):
        centers = self.HardTopK(self.window_size, label_probs) # [bs, window_size]
        centers = centers/(self.frame_num-1) # [bs, window_size]
        centers = rearrange(centers, "b w -> (b w)") # [bs*window_size]
        gaussians = self.generate_gauss_weight(centers, torch.tensor([self.width for i in range(centers.shape[0])]).to(self.device))
        gaussians = rearrange(gaussians, "(b w) t -> b w t", w=self.window_size) # [bs, window_size, frame_num]
        gaussians = torch.sum(gaussians, dim=1) # [bs, frame_num]
        gaussians = gaussians/gaussians.max(dim=-1, keepdim=True)[0] # [bs, frame_num]
        return gaussians

    def spatial_augmented(self, spatial_image_embeds, video_query_tokens, Qformer_atts, text_Qformer):
        bs = spatial_image_embeds.shape[0] // self.frame_num
        spatial_query_tokens = self.query_tokens.expand(bs, -1, -1) # [bs, 32, 768]
        spatial_image_embeds = rearrange(spatial_image_embeds, "(b t) n d -> b (t n) d", t=self.frame_num) # [bs, 257*frame_count, 1408]
        spatial_image_atts = torch.ones(spatial_image_embeds.size()[:-1], dtype=torch.long).to(spatial_image_embeds.device) # [bs, 257*frame_count] 
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask = Qformer_atts,
                query_embeds=spatial_query_tokens,
                encoder_hidden_states=spatial_image_embeds,
                encoder_attention_mask=spatial_image_atts,
                return_dict=True,
            )
        spatial_query_tokens = query_output.last_hidden_state[:,:spatial_query_tokens.size(1),:] # [bs, 32, 768]
        video_query_tokens = torch.cat([video_query_tokens, spatial_query_tokens], dim=1) # [bs, (4+1)*32, 768]
        return video_query_tokens
    
    def uniform_concat(self, samples):
        if self.use_vit:
            pixel_values = samples["pixel_values"] # [bs, frame_num, 3, 224, 224]
            bs, framecount, _, _, _ = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b t c h w -> (b t) c h w")
            frame_features_wo_norm = self.vision_model(pixel_values=pixel_values).last_hidden_state_without_norm
            frame_features_wo_norm = rearrange(frame_features_wo_norm, "(b t) n d -> b t n d", t=self.frame_num) # [bs, frame_num, 257, 1408] 
        else:
            frame_features_wo_norm = samples["frame_features"] # [bs, frame_num, 257, 1408]
        bs, framecount, _, _ = frame_features_wo_norm.shape
        frame_features_wo_norm = rearrange(frame_features_wo_norm, "b t n d -> (b t) n d") # [bs*frame_num, 257, 1408] 
        image_embeds = self.eva_vit_post_layer_norm(frame_features_wo_norm) # [bs*frame_num, 257, 1408]    
        spatial_image_embeds = image_embeds # [bs*frame_num, 257, 1408]

        if self.window_size < self.frame_num:
            image_embeds = rearrange(image_embeds, "(b t) n d -> b t n d", t=self.frame_num) # [bs, frame_count, 257, 1408] 
            def generate_uniform_elements(T, W):
                return torch.linspace(0, T-1, W, dtype=torch.int)
            indicators = generate_uniform_elements(self.frame_num, self.window_size).repeat(bs, 1) # [bs, window_size]
            selection_mask = torch.zeros(bs, self.window_size, self.frame_num).to(self.device) # [bs, window_size, frame_num]
            for i in range(bs):
                for j in range(self.window_size):
                    selection_mask[i][j][indicators[i][j]] = 1
            image_embeds = torch.einsum("b k t, b t n d -> b k n d", selection_mask, image_embeds) # [bs, window_size, 257, 1408]
            image_embeds = rearrange(image_embeds, "b w n d -> (b w) n d") # [bs*window_size, 257, 1408] 

        query_tokens = self.query_tokens.expand(bs*self.window_size, -1, -1) # [bs*window_size, 32, 768]
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(image_embeds.device) # [bs, 32]
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) # [bs, 32+seq_len]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids.repeat(self.window_size, 1),
                attention_mask = Qformer_atts.repeat(self.window_size, 1),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_tokens = query_output.last_hidden_state[:,:query_tokens.size(1),:] # [bs*window_size, 32, 768]
            video_query_tokens = rearrange(query_tokens, "(b t) n d -> b (t n) d", t=self.window_size) # [bs, window_size*32, 768]

            if self.use_spatial:
                video_query_tokens = self.spatial_augmented(spatial_image_embeds, video_query_tokens, Qformer_atts, text_Qformer)
            inputs_llm = self.t5_proj(video_query_tokens)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(inputs_llm.device) 

        regression_loss = torch.tensor(0).to(self.device)
        infoNCE_loss = torch.tensor(0).to(self.device)

        return inputs_llm, atts_llm, regression_loss, infoNCE_loss

    def oracle_concat(self, samples):
        if self.use_vit:
            pixel_values = samples["pixel_values"] # [bs, frame_count, 3, 224, 224]
            bs, framecount, _, _, _ = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b t c h w -> (b t) c h w")
            frame_features_wo_norm = self.vision_model(pixel_values=pixel_values).last_hidden_state_without_norm
            frame_features_wo_norm = rearrange(frame_features_wo_norm, "(b t) n d -> b t n d", t=framecount) # [bs, frame_count, 257, 1408] 
        else:
            frame_features_wo_norm = samples["frame_features"] # [bs, frame_count, 257, 1408]
        bs, framecount, _, _ = frame_features_wo_norm.shape
        frame_features_wo_norm = rearrange(frame_features_wo_norm, "b t n d -> (b t) n d") # [bs*frame_count, 257, 1408] 
        image_embeds = self.eva_vit_post_layer_norm(frame_features_wo_norm) # [bs*frame_count, 257, 1408] 
        spatial_image_embeds = image_embeds

        image_embeds_for_selection = self.branch_encoder.encode_image(frame_features_wo_norm) # [bs*frame_count, 1024]
        image_embeds_for_selection = rearrange(image_embeds_for_selection, "(b t) d -> b t d", t=framecount) # [bs, frame_count, 1024]
        label_embeds_for_selection = self.branch_encoder.encode_text(clip_text_tokenizer(samples["answers_text"]).to(image_embeds.device)) # [bs，1024] # answers_text, questions   
        def l2_norm(x):
            return x/x.norm(dim=-1, keepdim=True)
        
        label_probs = torch.einsum("b t d, b d -> b t", l2_norm(image_embeds_for_selection), l2_norm(label_embeds_for_selection))
        image_embeds = rearrange(image_embeds, "(b t) n d -> b t n d", t=framecount) # [bs, frame_count, 257, 1408] 

        label_probs = self.return_gmm_scores(label_probs)
        indicators = self.HardTopK(self.window_size, label_probs) # [bs, window_size]
        selection_mask = torch.zeros(bs, self.window_size, framecount).to(self.device) # [bs, window_size, frame_num]
        for i in range(bs):
            for j in range(self.window_size):
                selection_mask[i][j][indicators[i][j]] = 1
        image_embeds = torch.einsum("b k t, b t n d -> b k n d", selection_mask, image_embeds) # [bs, window_size, 257, 1408]

        image_embeds = rearrange(image_embeds, "b t n d -> (b t) n d") # [bs*4, 257, 1408] 
        query_tokens = self.query_tokens.expand(bs*self.window_size, -1, -1) # [bs*frame_count, 32, 768]
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(image_embeds.device) # [bs, 32]
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) # [bs, 32+seq_len]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids.repeat(self.window_size, 1),
                attention_mask = Qformer_atts.repeat(self.window_size, 1),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_tokens = query_output.last_hidden_state[:,:query_tokens.size(1),:] # [bs*frame_count, 32, 768]
            query_tokens = rearrange(query_tokens, "(b w) n d -> b w n d", w=self.window_size) # [bs, frame_count, 32, 768]

            position_ids = torch.arange(self.window_size, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(bs, -1)
            frame_embedding = self.frame_embeds(position_ids)
            frame_embedding = frame_embedding.unsqueeze(-2)
            query_tokens = query_tokens + frame_embedding # [bs, frame_count, 32, 768]
            video_query_tokens = rearrange(query_tokens, "b w n d -> b (w n) d") # [bs, 4*32, 768]

            if self.use_spatial:
                video_query_tokens = self.spatial_augmented(spatial_image_embeds, video_query_tokens, Qformer_atts, text_Qformer)
            inputs_llm = self.t5_proj(video_query_tokens)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(inputs_llm.device) 

        regression_loss = torch.tensor(0).to(self.device)
        infoNCE_loss = torch.tensor(0).to(self.device)

        return inputs_llm, atts_llm, regression_loss, infoNCE_loss   
    
    def grounding_concat(self, samples):
        if self.use_vit:
            pixel_values = samples["pixel_values"] # [bs, frame_count, 3, 224, 224]
            bs, framecount, _, _, _ = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b t c h w -> (b t) c h w")
            frame_features_wo_norm = self.vision_model(pixel_values=pixel_values).last_hidden_state_without_norm
            frame_features_wo_norm = rearrange(frame_features_wo_norm, "(b t) n d -> b t n d", t=framecount) # [bs, frame_count, 257, 1408] 
        else:
            frame_features_wo_norm = samples["frame_features"] # [bs, frame_count, 257, 1408]
        bs, framecount, _, _ = frame_features_wo_norm.shape
        frame_features_wo_norm = rearrange(frame_features_wo_norm, "b t n d -> (b t) n d") # [bs*frame_count, 257, 1408] 
        image_embeds = self.eva_vit_post_layer_norm(frame_features_wo_norm) # [bs*frame_count, 257, 1408] 
        spatial_image_embeds = image_embeds

        image_embeds_for_selection = self.branch_encoder.encode_image(frame_features_wo_norm) # [bs*frame_count, 1024]
        image_embeds_for_selection = rearrange(image_embeds_for_selection, "(b t) d -> b t d", t=framecount) # [bs, frame_count, 1024]
        label_embeds_for_selection = self.branch_encoder.encode_text(clip_text_tokenizer(samples["answers_text"]).to(image_embeds.device)) # [bs，1024]

        if 'options_a0' not in samples.keys():
            question_embeds_for_selection = self.branch_encoder.encode_text(clip_text_tokenizer(samples["questions"]).to(image_embeds.device)) # [bs，1024]
            question_embeds_for_selection = question_embeds_for_selection.unsqueeze(1) # [bs，1, 1024]
        else:
            question_embeds_for_selection = self.branch_encoder.encode_text(clip_text_tokenizer(samples["questions"]).to(image_embeds.device)) # [bs，1024]
            options_a0 = self.branch_encoder.encode_text(clip_text_tokenizer(samples["options_a0"]).to(image_embeds.device)) # [bs，1024]
            options_a1 = self.branch_encoder.encode_text(clip_text_tokenizer(samples["options_a1"]).to(image_embeds.device)) # [bs，1024]
            if 'options_a2' not in samples.keys():
                question_embeds_for_selection = torch.stack([question_embeds_for_selection,options_a0,options_a1], dim=1) # [bs，3, 1024]
            else:
                options_a2 = self.branch_encoder.encode_text(clip_text_tokenizer(samples["options_a2"]).to(image_embeds.device)) # [bs，1024]
                options_a3 = self.branch_encoder.encode_text(clip_text_tokenizer(samples["options_a3"]).to(image_embeds.device)) # [bs，1024]
                if 'options_a4' not in samples.keys():
                    question_embeds_for_selection = torch.stack([question_embeds_for_selection,options_a0,options_a1,options_a2,options_a3], dim=1) # [bs，5, 1024]
                else:
                    options_a4 = self.branch_encoder.encode_text(clip_text_tokenizer(samples["options_a4"]).to(image_embeds.device)) # [bs，1024]
                    question_embeds_for_selection = torch.stack([question_embeds_for_selection,options_a0,options_a1,options_a2,options_a3,options_a4], dim=1) # [bs，6, 1024]

        def l2_norm(x):
            return x/x.norm(dim=-1, keepdim=True)
        
        label_probs = torch.einsum("b t d, b d -> b t", l2_norm(image_embeds_for_selection), l2_norm(label_embeds_for_selection))
        image_embeds = rearrange(image_embeds, "(b t) n d -> b t n d", t=framecount) # [bs, frame_count, 257, 1408] 
        image_embeds, regression_loss, infoNCE_loss = self.grounding(Q=question_embeds_for_selection, K=image_embeds_for_selection, V=image_embeds, answer_embeds=label_embeds_for_selection, label_probs=label_probs, answers_id=samples["answers_id"].to(self.device) if 'answers_id' in samples.keys() else None) # [bs, 4, 257, 1408]

        image_embeds = rearrange(image_embeds, "b w n d -> (b w) n d") # [bs*4, 257, 1408] 
        query_tokens = self.query_tokens.expand(bs*self.window_size, -1, -1) # [bs*frame_count, 32, 768]
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(image_embeds.device) # [bs, 32]
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) # [bs, 32+seq_len]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids.repeat(self.window_size, 1),
                attention_mask = Qformer_atts.repeat(self.window_size, 1),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_tokens = query_output.last_hidden_state[:,:query_tokens.size(1),:] # [bs*frame_count, 32, 768]
            query_tokens = rearrange(query_tokens, "(b w) n d -> b w n d", w=self.window_size) # [bs, frame_count, 32, 768]

            position_ids = torch.arange(self.window_size, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(bs, -1)
            frame_embedding = self.frame_embeds(position_ids)
            frame_embedding = frame_embedding.unsqueeze(-2)
            query_tokens = query_tokens + frame_embedding # [bs, frame_count, 32, 768]
            video_query_tokens = rearrange(query_tokens, "b w n d -> b (w n) d") # [bs, 4*32, 768]

            if self.use_spatial:
                video_query_tokens = self.spatial_augmented(spatial_image_embeds, video_query_tokens, Qformer_atts, text_Qformer)
            inputs_llm = self.t5_proj(video_query_tokens)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(inputs_llm.device) 

        return inputs_llm, atts_llm, regression_loss, infoNCE_loss   
    
    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        if self.mode == 'grounding':
            inputs_t5, atts_t5, regression_loss, infoNCE_loss = self.grounding_concat(samples)
        elif self.mode == 'uniform':
            inputs_t5, atts_t5, regression_loss, infoNCE_loss = self.uniform_concat(samples)
        elif self.mode == 'oracle':
            inputs_t5, atts_t5, regression_loss, infoNCE_loss = self.oracle_concat(samples)

        with self.maybe_autocast():
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_input_txt_len,
                return_tensors="pt",
            ).to(inputs_t5.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(inputs_t5.device)
            
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            vqa_loss = outputs.loss

            return {
                "loss": vqa_loss+regression_loss+infoNCE_loss,
                "vqa_loss": vqa_loss,
                "regression_loss": regression_loss,
                "infoNCE_loss": infoNCE_loss,
                }

    @torch.no_grad()
    def generate(
        self,
        samples,
        **generate_kwargs
    ):

        if self.mode == 'grounding':
            inputs_t5, atts_t5, _, _ = self.grounding_concat(samples)
        elif self.mode == 'uniform':
            inputs_t5, atts_t5, _, _ = self.uniform_concat(samples)
        elif self.mode == 'oracle':
            inputs_t5, atts_t5, _, _ = self.oracle_concat(samples)

        with self.maybe_autocast():
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                return_tensors="pt"
            ).to(inputs_t5.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                **generate_kwargs
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

