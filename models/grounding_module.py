import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import sys
import os
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import sys
import math
import os
from transformers.activations import gelu
from transformers.modeling_outputs import BaseModelOutput
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
import torch.nn.functional as F
from models.Transformer import Transformer, Embeddings
import torch.distributed as dist

class TokenTypeEmbeddings(nn.Module):
    def __init__(
        self, d_model, token_type_num
    ):
        super().__init__()
        self.modality_embedding = nn.Embedding(token_type_num, d_model)
        self.type2id = {'question': 0, 'video': 1, 'object': 2}
        nn.init.zeros_(self.modality_embedding.weight)

    def forward(self, embeddings, token_type):
        seq_length = embeddings.size(1)
        token_type_id = self.type2id[token_type]
        modality_embeddings = self.modality_embedding(torch.tensor([token_type_id] * seq_length, dtype=torch.long).to(embeddings.device))
        return modality_embeddings

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)
    
class Grounding(nn.Module):
    def __init__(self, dim=1024, heads=4, dropout=0.3, window_size=4, frame_num=32, width=0.15, temperature=0.1):
        super(Grounding, self).__init__()

        self.dtype = torch.float32
        self.frame_num = frame_num
        self.window_size = window_size
        self.width = width
        self.temperature = temperature
        self.sigma = 9
        self.mult = 4
        self.num_hidden_layers = 2
        self.inner_dim = dim // 4
        self.heads = heads
        self.activation = 'gelu'
        self.use_proj = True

        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, self.inner_dim, bias=False),
            )
        self.modality_embedding = TokenTypeEmbeddings(d_model=self.inner_dim, token_type_num=2)
        self.position_v = Embeddings(self.inner_dim, 0, self.frame_num, dropout, True)

        self.encoder = Transformer(
            AutoConfig.from_pretrained(
            "FacebookAI/roberta-base", 
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.inner_dim,
            attention_probs_dropout_prob=dropout,
            intermediate_size=self.mult*self.inner_dim,
            num_attention_heads=self.heads,
            hidden_act = self.activation,
            ))
        self.satt_pool_frame = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim // 2),
                nn.Tanh(),
                nn.Linear(self.inner_dim // 2, 1),
                nn.Softmax(dim=-2)) 
        self.logit_gauss_c = nn.Sequential(
                                nn.Dropout(dropout), 
                                nn.Linear(self.inner_dim, self.window_size),
                                nn.Sigmoid(),
                                )     
        if self.use_proj:   
            self.proj_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, self.inner_dim // 2, bias=False),
                nn.ReLU() if self.activation == 'relu' else nn.GELU(),
                nn.Linear(self.inner_dim // 2, self.inner_dim // 2)
                )
   
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
    
    def HardTopK(self, k, x):
        topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, k
        indices = torch.sort(indices, dim=-1).values
        return indices

    def get_indicator(self, scores, k, sigma=0.05):
        indicator = PerturbedTopKFunction.apply(scores, k, 500, sigma)
        return indicator
        
    def calculate_kl_divergence(self, pred_scores, label_scores):
        """
        scores: [bs, option_num]
        labels: [bs]
        """
        pred_scores = F.log_softmax(pred_scores, dim=-1)
        label_scores = F.softmax(label_scores, dim=-1)
        kl_divergence = F.kl_div(pred_scores, label_scores, reduction='batchmean')
        return kl_divergence

    def mmt_encode(self, video_embeds, text_embeds, gaussian_weight=None):
        """
        video_embeds: [bs, frame_num, dim]
        text_embeds: [bs, seq, dim]
        gaussian_weight: [bs, frame_num] / None
        """
        bs = video_embeds.shape[0]
        video_embeds = self.position_v(self.embedding(video_embeds))
        text_embeds = self.embedding(text_embeds)

        input_embeds = torch.cat([video_embeds, text_embeds], dim=1)
        input_embeds[:,:self.frame_num,:] = input_embeds[:,:self.frame_num,:] + self.modality_embedding(input_embeds[:,:self.frame_num,:], "video")
        input_embeds[:,self.frame_num:,:] = input_embeds[:,self.frame_num:,:] + self.modality_embedding(input_embeds[:,self.frame_num:,:], "question")
        hidden_states = self.encoder(x=input_embeds,
                                attn_mask=torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(self.device),
                                gauss_weight=gaussian_weight)[0] # [bs, frame_num+seq_len, dim]   
        hidden_states = hidden_states[:,:self.frame_num,:] # [bs, frame_num, dim] 
        hidden_states = self.position_v(hidden_states) 
        fatt_gs = self.satt_pool_frame(hidden_states) # [bs, frame_num, 1]
        pooled_qv_feat = torch.sum(hidden_states*fatt_gs, dim=1) # [bs, dim] 

        gauss_c = self.logit_gauss_c(pooled_qv_feat) # [bs, window_size]
        gauss_w = torch.full((gauss_c.shape[0], self.window_size), self.width).to(self.device) # [bs, window_size]
        pred_gaussians = self.generate_gmm_weight(gauss_c, gauss_w) # [bs, frame_num]

        return pred_gaussians, gauss_c, gauss_w                              

    def generate_gauss_weight(self, center, width):
        # code copied from https://github.com/minghangz/cpl
        weight = torch.linspace(0, 1, self.frame_num)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma        
        w = 0.3989422804014327 #1/(math.sqrt(2*math.pi))
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        return weight
        # return weight/weight.max(dim=-1, keepdim=True)[0]
    
    def generate_gmm_weight(self, centers, widths):
        """
        centers: [bs, window_size]
        """
        centers = rearrange(centers, "b w -> (b w)") # [bs*window_size]
        widths = rearrange(widths, "b w -> (b w)") # [bs*window_size]
        gaussians = self.generate_gauss_weight(centers, widths)
        gaussians = rearrange(gaussians, "(b w) t -> b w t", w=self.window_size) # [bs, window_size, frame_num]
        gaussians = torch.sum(gaussians, dim=1) # [bs, frame_num]
        gaussians = gaussians/gaussians.max(dim=-1, keepdim=True)[0] # [bs, frame_num]   
        return gaussians    

    def calculate_ce_loss(self, scores, labels):
        """
        scores: [bs, option_num]
        labels: [bs]
        """
        loss = nn.CrossEntropyLoss()(scores, labels)
        preds = torch.argmax(scores, dim=-1)
        return loss, preds
    
    def calculate_regression_loss(self, pred_center, label_probs):
        """
        pred_center: [bs, window_size]
        label_probs: [bs, frame_num]
        """
        pred_center = pred_center*(self.frame_num-1)
        positive_center = self.HardTopK(self.window_size, label_probs)
        zero = torch.tensor(0.0).to(self.device)
        one = torch.tensor(1.0).to(self.device)

        oreder_loss = torch.tensor(0.0).to(self.device)
        for i in range(pred_center.shape[0]):
            temp_loss = torch.tensor(0.0).to(self.device)
            for j in range(self.window_size-1):
                temp_loss += torch.max(one + pred_center[i,j] - pred_center[i,j+1], zero) 
            oreder_loss += temp_loss/(self.window_size-1)
        oreder_loss = oreder_loss/(pred_center.shape[0])
        loss_bbox = F.smooth_l1_loss(pred_center, positive_center)
        return loss_bbox + oreder_loss
    
    def calculate_contrastive_loss(self, video_embeds, options_embeds, answer_embeds, pred_gaussians, answers_id):
        """
        video_embeds: [bs, frame_num, dim]
        options_embeds: [bs, 5, dim]
        answer_embeds: [bs, dim]
        label_probs: [bs, frame_num]
        pred_gaussians: [bs, frame_num]
        answers_id: [bs]
        """
        bs = video_embeds.shape[0]
        intra_neg_num = self.frame_num - self.window_size
        inter_neg_num_per_video = self.window_size
        inter_neg_num = (bs-1)*inter_neg_num_per_video

        # pred_video_embeds 
        pred_video_embeds = torch.einsum("b k t, b t d -> b k d", self.get_indicator(pred_gaussians, self.window_size), video_embeds) # [bs, window_size, dim]
        # intra_negative_video_embeds 
        negative_gaussians = torch.tensor(1).to(self.device)-pred_gaussians
        intra_negative_video_embeds = torch.einsum("b k t, b t d -> b k d", self.get_indicator(negative_gaussians, intra_neg_num), video_embeds) # [bs, intra_neg_num, dim]
        # inter_negative_video_embeds 
        def del_element(index, x):
            return torch.cat((x[:index], x[index+1:]))    
        shuffle_video_embeds = video_embeds[:,torch.randperm(self.frame_num),:][:,:inter_neg_num_per_video,:] # [bs, inter_neg_num_per_video, dim] 打乱第二个维度
        inter_negative_video_embeds = []
        for i in range(bs):
            temp = del_element(i, shuffle_video_embeds) # [bs-1, inter_neg_num_per_video, dim]
            temp = rearrange(temp, "b w o -> (b w) o") # [(bs-1)*inter_neg_num_per_video, dim]
            inter_negative_video_embeds.append(temp)
        inter_negative_video_embeds = torch.stack(inter_negative_video_embeds, dim=0) # [bs, (bs-1)*inter_neg_num_per_video, dim]
        # inter_negative_answer_embeds 
        inter_negative_answer_embeds = []
        for i in range(bs):
            temp = del_element(i, answer_embeds) # [bs-1, dim]
            inter_negative_answer_embeds.append(temp)
        inter_negative_answer_embeds = torch.stack(inter_negative_answer_embeds, dim=0) # [bs, (bs-1), dim]
        # options_negative_embeds 
        if options_embeds != None:
            options_negative_embeds = []
            for i in range(bs):
                temp = []
                for j in range(options_embeds.shape[1]):
                    if j != answers_id[i]:
                        temp.append(options_embeds[i,j])
                temp = torch.stack(temp, dim=0).to(self.device) # [4, dim]
                options_negative_embeds.append(temp)
            options_negative_embeds = torch.stack(options_negative_embeds, dim=0) # [bs, 4, dim]

        # proj_head
        if self.use_proj:
            pred_video_embeds = self.proj_head(pred_video_embeds)
            intra_negative_video_embeds = self.proj_head(intra_negative_video_embeds)
            inter_negative_video_embeds = self.proj_head(inter_negative_video_embeds)
            answer_embeds = self.proj_head(answer_embeds)
            inter_negative_answer_embeds = self.proj_head(inter_negative_answer_embeds)
            if options_embeds != None:
                options_negative_embeds = self.proj_head(options_negative_embeds)

        def l2_norm(x):
            return x/x.norm(dim=-1, keepdim=True)
        # compute positive_logits
        positive_logits = torch.einsum("b w d, b d -> b w", l2_norm(pred_video_embeds), l2_norm(answer_embeds)) # [bs, window_size]
        # compute intra_negative_video_logits
        intra_negative_video_logits = torch.einsum("b w d, b d -> b w", l2_norm(intra_negative_video_embeds), l2_norm(answer_embeds)) # [bs, intra_neg_num]
        # compute inter_negative_video_logits
        inter_negative_video_logits = torch.einsum("b w d, b d -> b w", l2_norm(inter_negative_video_embeds), l2_norm(answer_embeds)) # [bs, inter_neg_num]
        # compute inter_negative_answer_logits
        inter_negative_answer_logits = torch.einsum("b w d, b o d -> b w o", l2_norm(pred_video_embeds), l2_norm(inter_negative_answer_embeds)) # [bs, window_size, bs-1]
        # compute option_negative_logits
        if options_embeds != None:
            options_negative_logits = torch.einsum("b w d, b o d -> b w o", l2_norm(pred_video_embeds), l2_norm(options_negative_embeds)) # [bs, window_size, 4]
        
        # compute infoNCE loss
        infoNCE_loss = 0
        labels = torch.zeros(bs, dtype=torch.long, device=self.device)
        for i in range(self.window_size):
            if options_embeds != None:
                logits = torch.cat([positive_logits[:,i].unsqueeze(-1), # [bs, 1]
                                    intra_negative_video_logits, # [bs, intra_neg_num]
                                    options_negative_logits[:, i, :], # [bs, 4]
                                    # inter_negative_video_logits, # [bs, inter_neg_num]
                                    inter_negative_answer_logits[:, i, :], # [bs, bs-1]
                                    ], dim=1)
            else:
                logits = torch.cat([positive_logits[:,i].unsqueeze(-1), # [bs, 1]
                                    intra_negative_video_logits, # [bs, intra_neg_num]
                                    # inter_negative_video_logits, # [bs, inter_neg_num]
                                    inter_negative_answer_logits[:, i, :], # [bs, bs-1]
                                    ], dim=1)
            infoNCE_loss += F.cross_entropy(logits/self.temperature, labels)
        infoNCE_loss = infoNCE_loss/self.window_size
        return infoNCE_loss

    def forward(self, Q, K, V, answer_embeds, label_probs=None, answers_id=None):
        """
        baseline:72.64
        oracle: 79.09
        """

        '''
        Q: [bs, seq_len, dim]
        K: [bs, frame_num, dim]
        V: [bs, frame_num, query_num, dim]
        answer_embeds: [bs, dim]
        label_probs: [bs, frame_num]
        answers_id: [bs]
        '''
        video_embeds = K # [bs, frame_num, dim]
        question_embeds = Q # [bs, seq_len, dim]
        if question_embeds.shape[1] > 1:
            option_embeds = Q[:,1:,:] # [bs, 5, dim]
        else:
            option_embeds = None
        answer_embeds = answer_embeds # [bs, dim]

        pred_gaussians, gauss_c, gauss_w = self.mmt_encode(video_embeds, question_embeds) # [bs, frame_num]

        # pos_centers = self.HardTopK(self.window_size, label_probs)/(self.frame_num-1)
        # pos_gaussians = self.generate_gmm_weight(pos_centers, gauss_w) # [bs, frame_num]
        # kl_loss = self.calculate_kl_divergence(pred_gaussians, pos_gaussians)
        regression_loss = self.calculate_regression_loss(gauss_c, label_probs)
        infoNCE_loss = self.calculate_contrastive_loss(video_embeds, option_embeds, answer_embeds, pred_gaussians, answers_id)
    
        if self.training:
            selection_mask = self.get_indicator(pred_gaussians, self.window_size) # [bs, window_size, frame_num]
            selected_V = torch.einsum("b k t, b t q d -> b k q d", selection_mask, V) # [bs, window_size, query_num, dim]
        else:
            indicators = self.HardTopK(self.window_size, pred_gaussians) # [bs, window_size]
            selection_mask = torch.zeros(K.shape[0], self.window_size, K.shape[1]).to(self.device) # [bs, window_size, frame_num]
            for i in range(K.shape[0]):
                for j in range(self.window_size):
                    selection_mask[i][j][indicators[i][j]] = 1
            selected_V = torch.einsum("b k t, b t q d -> b k q d", selection_mask, V) # [bs, window_size, query_num, dim]
        return selected_V, regression_loss, infoNCE_loss

# bs = 10    
# model = Grounding(window_size=4, dim=1024, heads=8, dropout=0.1, frame_num=32, width=0.2, temperature=0.1)
# print(get_parameter_number(model))
# model(Q=torch.randn(bs, 6, 1024), 
#       K=torch.randn(bs, 32, 1024), 
#       V=torch.randn(bs, 32, 257, 1408), 
#       answer_embeds=torch.randn(bs, 1024), 
#       label_probs=torch.randn(bs, 32), 
#       answers_id=torch.tensor([0 for i in range(bs)]))
