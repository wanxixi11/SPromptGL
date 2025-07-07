import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from layers import *
from utils import *
from CLIP.clip import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
attr_words = [
                "Nromal brain volume","Uniform cortical thickness", "Symmetrical brain structure", "Normal ventricular morphology",
                "Altered brain volume", "Abnormal functional connectivity", "White matter integrity issues","Different activation patterns"
                ]

text = clip.tokenize(attr_words).to(device)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.t1 = nn.Linear(input_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        return x

class PLN(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(PLN, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        # self.input_data_dims = [x * 2 for x in self.input_data_dims]
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        self.fusion_weight = nn.Parameter(torch.ones([6,1]))
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.device = "cuda"
        self.ViT_model, preprocess = clip.load('RN50', self.device)
        self.MLP = MLP(input_dim = 4096,hidden_dim = 128, output_dim = 9)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        combine_part = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
        with torch.no_grad():
            text_features = self.ViT_model.encode_text(text).to(device).float()
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
        
        
        part_x = x.view(871, 4, 4, 9)
        part_x_norm = F.normalize(part_x, p=2, dim=-1)
        text_features_norm = F.normalize(text_features, p=2, dim=-1)

        # 扩展 text_features_norm 的维度以便与 part_x_norm 进行广播
        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # 变为 [1, 1, 1, 2, 9]
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)  # 变为 [871, 4, 4, 1, 9]

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))  # 结果为 [871, 4, 4, 1, 2]
        similarity_matrix = similarity_matrix[:,:,:,-1,:]

        # 进行数值截断以避免数值问题
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)

        # 计算 softmax
        similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)

        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
        # 对最后一个维度进行平均池化
        
        # 需要先将输入张量的形状调整为 (871*4, 1, 36) 以适应 AvgPool1d 的输入格式
        reshaped_x = x.view(-1, 1, 36)
        global_x = self.avg_pool(reshaped_x)
        m = part_x.shape[2]

        # 重新调整张量形状为 (871, 4, 9)
        global_x = global_x.view(871, 4, 9)
        for i in range(m):
            part = part_x[:,:,i,:]
            Vsim = F.softmax(part*global_x, dim=-1) 
            part_embedding = Vsim*part
            combine_part.append(part_embedding)
        x = x + torch.cat(combine_part,dim=2)
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1) 
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
  
class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode = 'Sigmoid-like'):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th    
    def forward(self, x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        th = self.th
        x = self.p(x)
        x_norm = F.normalize(x,dim=-1)
        score = torch.matmul(x_norm, x_norm.T)
        mask = (score > th).detach().float()
        markoff_value = 0
        output = score * mask + markoff_value * (1 - mask)
        return output  
    
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(input_dim, output_dim)
        self.gcn2 = GraphConv(input_dim, output_dim)
        
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x
    


# 获取权重矩阵

def adj_normalization(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

    
class GFR(nn.Module):
    def __init__(self, input_dim , output_dim ,th , mode='Sigmoid-like', label_drop = 0.,proj_drop =0.):
        super(GFR, self).__init__()
        self.th = th
        self.mode = mode
        # breakpoint()
        
        input_sum = sum(input_dim) 
    
        self.fusion_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        self.combined_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        
        self.GraphConstruct0 = GraphLearn(input_dim = input_dim[0],th = self.th, mode = self.mode)
        self.MessagePassing0 = GCN (input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct1 = GraphLearn(input_dim = input_dim[1], th = self.th, mode = self.mode)
        self.MessagePassing1 = GCN(input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct2 = GraphLearn(input_dim = input_dim[2], th = self.th, mode = self.mode)
        self.MessagePassing2 = GCN(input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct3 = GraphLearn(input_dim = input_dim[3], th = self.th, mode = self.mode)
        self.MessagePassing3 = GCN(input_dim= input_sum, output_dim= input_sum)

        self.proj_drop = nn.Dropout(proj_drop)
        self.label_drop = nn.Dropout(label_drop)


    def forward(self, x, model_num, label, index):
        combined_feat = []
        all_adj = []
        N = len(x[0])
        new_label = label.clone()
        new_label[new_label == 0] = -1
        new_label = [x if i in index else 0 for i, x in enumerate(new_label)]
        new_label = torch.tensor(new_label)
        l1 = new_label.unsqueeze(1).repeat(1, N)  
        l2 = new_label.t().repeat(N, 1)  
        adj1 = l1.eq_(l2)
        A = normalize_adj(adj1.cuda() + torch.eye(adj1.size(0)).cuda())
 
        for i in range(model_num):  
            f_w = self.fusion_weight[i]
            f_w = torch.sigmoid(f_w)
            c_w = self.combined_weight[i]
            c_w = torch.sigmoid(c_w)
            
            if i==0:
                adj = self.GraphConstruct0(x[i])
                normalized_adj0 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj0 - A).to(f_w.device)
                feat0 = self.MessagePassing0(torch.cat(x,dim=1), normalized_adj0)
                combined_feat = f_w*feat0+torch.cat(x,dim=1)

            elif i==1:
                adj = self.GraphConstruct1(x[i])
                normalized_adj1 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj1 - A).to(f_w.device) + ALoss
                feat1 = self.MessagePassing1(torch.cat(x,dim=1), normalized_adj1)
                combined_feat = f_w*feat1+torch.cat(x,dim=1)
                
            elif i==2:
                adj = self.GraphConstruct2(x[i])
                normalized_adj2 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj2 - A).to(f_w.device) + ALoss
                feat3 = self.MessagePassing2(torch.cat(x,dim=1), normalized_adj2)
                combined_feat = f_w*feat3+combined_feat

                        
            elif i==3:
                adj = self.GraphConstruct3(x[i])
                normalized_adj3 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj3 - A).to(f_w.device) + ALoss
                feat3 = self.MessagePassing3(torch.cat(x,dim=1), normalized_adj3)
                combined_feat = f_w*feat3+combined_feat
            all_adj.append(adj)
                   
        return combined_feat , all_adj, ALoss/model_num
    
    
        