import argparse
import numpy as np
import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input = input.float()
        # adj = adj.float()
        support = input @ self.weight
        # support = torch.tensor(support, dtype=torch.double)
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class EdgeEncoding(nn.Module):
    def __init__(self, config):
        super(EdgeEncoding, self).__init__()
        self.config = config

        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index,
                                                config.hidden_size)  # 输入向量的最后一维不能超过config.max_inti_pos_index  向量增加一个维度 最后一维为config.hidden_size
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None):
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.hop_dis_embeddings(time_dis_ids)

        embeddings = position_embeddings + hop_embeddings + time_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyModel(nn.Module):
    def __init__(self, args,data, use_atten = True):
        super(MyModel, self).__init__()
        self.args = args
        self.data = data
        self.use_atten = use_atten
        self.gcn = GCN_Model(args)
        self.gru = nn.GRU(self.args.out_dim, self.args.gru_dim)
        self.attention = Self_Attenyion_Layer(self.args)
        self.temp_attention = TemporalAttention(self.args)
        self.score = Score(self.args)

    def mean_pools(self, x):
        res = torch.sum(x, dim=1) / x.size()[-2]
        return res


    def forward(self, x, subgraph_adj, snap, negative_data):

        if self.args.mask:
            mask = torch.ones_like(x);
            # 计算每个时间步需要置为 0 的节点数
            num_zero_nodes = int(x.size()[1] * 0.05)

            for t in range(x.size()[0]):
                # 随机选择 5% 的节点索引
                zero_indices = torch.randperm(x.size()[1])[:num_zero_nodes]
                for idx in zero_indices:
                    mask[t, idx, :] = 0  # 将选择的节点特征全部置为 0
            # 应用掩码到 tensor
            x = x * mask

        x = self.gcn(x,subgraph_adj)  # (times, node_num,sn, embedding_dim)
        if self.args.use_dif:
            x = x[:,:,0,:]  # (times, node_num, embedding_dim)
        if self.args.res:
            x1 = x.permute(1, 0, 2)  #0.3  0.35
            #x1 = self.attention(x1)

        # x = x.permute(1,0,2) #(node_num, times, embedding_dim)
        x, _ = self.gru(x)  # (times,node_num,gru_dim)
        # self.h_0= x[1,:,:].unsqueeze(dim=0)
        # if snap <= self.args.window_size-1:
        #     x = torch.stack([x[:,i,:] for i in range(snap+1)],dim=1)
        # else:
        #     x = x[:,snap-self.args.window_size:snap,:] # (node_num, snap-l+1:snap,attention_hidden_size)
        x = x.permute(1, 0, 2)  # (node_num, times, gru_dim)
        if self.use_atten:
            x = self.attention(x)
            if self.args.res:
                #x = torch.cat((x, x1), 1)
                x = x + x1
            x = self.mean_pools(x) # torch.Size([1899, 64])
        else:
            x = x[:,-1,:]
        # x = x[:,-1,:]
        if negative_data:
            edges = torch.cat((torch.Tensor(self.data['edges'][snap]), negative_data[snap]), dim=0).T  # [2,2000]
            hi = x[edges[0].numpy().tolist()]
            hj = x[edges[1].numpy().tolist()]
        else:
            edges = torch.Tensor(self.data['edges'][snap]).T
            hi = x[edges[0].numpy().tolist()]
            hj = x[edges[1].numpy().tolist()]

        score = self.score(hi, hj)

        return score


class GCN_Model(nn.Module):
    def __init__(self, args):
        super(GCN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = GraphConvolution(args.in_dim, args.hidden_dim)
        self.gcn2 = GraphConvolution(args.hidden_dim, args.out_dim)

    def forward(self, x, adj):
        x_ = F.relu(self.gcn1(x, adj))
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x_ = self.gcn2(x_, adj)
        x_ = F.relu(x_)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        return x_


class TemporalAttention(nn.Module):
    def __init__(self, args):
        super(TemporalAttention, self).__init__()
        self.args = args
        self.input_size = args.gru_dim
        self.hidden_size = int(self.args.attention_hidden_size / self.args.num_attention_heads)
        self.num_heads = args.num_attention_heads

        # 定义注意力计算的权重矩阵
        self.W_Q = nn.Linear(self.input_size, self.hidden_size * self.num_heads, bias=False)
        self.W_K = nn.Linear(self.input_size, self.hidden_size * self.num_heads, bias=False)
        self.W_V = nn.Linear(self.input_size, self.hidden_size * self.num_heads, bias=False)
        self.W_O = nn.Linear(self.hidden_size * self.num_heads, self.args.attention_hidden_size, bias=False)

        self.scale_factor = self.hidden_size ** 0.5  # 缩放因子

    def forward(self, x):
        # 输入 x 的维度为 (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # 将输入进行线性变换得到 Q、K、V
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(scores, dim=-1)

        # 加权池化得到加权表示
        weighted_sum = torch.matmul(attention_weights, V)
        weighted_sum = weighted_sum.view(batch_size, seq_len, -1)

        # 输出表示信息
        output = self.W_O(weighted_sum)

        return output


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for channel attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Calculate global average pooling
        avg_pool = self.avg_pool(x)

        # Flatten the feature map
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        # Pass through the fully connected layers
        channel_attention = self.fc(avg_pool)

        # Apply channel attention to the input feature map
        channel_attention = channel_attention.view(channel_attention.size(0), channel_attention.size(1), 1, 1)
        x = x * channel_attention

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for channel attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Calculate global average pooling
        avg_pool = self.avg_pool(x)

        # Flatten the feature map
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        # Pass through the fully connected layers
        channel_attention = self.fc(avg_pool)

        # Apply channel attention to the input feature map
        channel_attention = channel_attention.view(channel_attention.size(0), channel_attention.size(1), 1, 1)
        x = x * channel_attention

        return x


class Self_Attenyion_Layer(nn.Module):
    def __init__(self, args):
        super(Self_Attenyion_Layer, self).__init__()
        self.all_hidden_size = args.attention_hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.attention_hidden_size = int(self.all_hidden_size / self.num_attention_heads)
        self.args = args
        self.query = nn.Linear(args.gru_dim, self.attention_hidden_size * self.num_attention_heads)
        self.key = nn.Linear(args.gru_dim, self.attention_hidden_size * self.num_attention_heads)
        self.value = nn.Linear(args.gru_dim, self.attention_hidden_size * self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_hidden_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, input_tensor):  # input_tensor(node_num, times,gru_dim)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_hidden_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = F.dropout(attention_probs, self.args.dropout, training=self.training)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)


        return context_layer


class Score(nn.Module):
    def __init__(self, args):
        super(Score, self).__init__()
        if args.use_atten:
            self.linear = nn.Sequential(nn.Linear(args.attention_hidden_size * 2, args.score_size),
                                        nn.ReLU(),
                                        nn.Linear(args.score_size, 1))
        else:
            self.linear = nn.Sequential(nn.Linear(args.gru_dim * 2, args.score_size),
                                        nn.ReLU(),
                                        nn.Linear(args.score_size, 1))
        # self.linear = nn.Sequential(nn.Linear(args.attention_hidden_size * 2, 1))
        self.dropout = args.dropout

    def forward(self, hi, hj):
        h = torch.cat((hi, hj), dim=1)
        s = self.linear(h)
        s = F.dropout(s, self.dropout, training=self.training)
        return s[:, 0]
