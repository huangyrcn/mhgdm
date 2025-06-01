"""Euclidean layers."""
import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act(hidden_dim, act_name, num_layers, enc=True, dim=None):
    """
    Helper function to get dimension and activation at every layer.
    :param hidden_dim: hidden dimension size
    :param act_name: activation function name (e.g., 'ReLU')
    :param num_layers: number of layers
    :param enc: whether this is for encoder (True) or decoder (False)
    :param dim: dimension for decoder (only used when enc=False)
    :return: dims, acts
    """
    act = getattr(nn, act_name)
    acts = [act()] * (num_layers)

    if enc:
        dims = [hidden_dim] * (num_layers+1) # len=args.num_layers+1
    else:
        dims = [dim]+[hidden_dim] * (num_layers)   # len=args.num_layers+1

    return dims, acts


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(), edge_dim=0, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        dense_att_init_edge_dim = 1  # From adj.unsqueeze(-1) as the third argument
        if self.edge_dim > 0:
            dense_att_init_edge_dim += 1 # From adj.unsqueeze(-1) as the fourth argument
        self.att = DenseAtt(out_dim, dropout, edge_dim=dense_att_init_edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
        self.act = act
        if msg_transform:
            self.msg_net = nn.Sequential(
                nn.Linear(out_dim+1, out_dim),
                act,
                nn.Linear(out_dim, out_dim)
            )
        if sum_transform:
            self.out_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                nn.Linear(out_dim, out_dim)
            )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input):
        x, adj = input
        x = self.linear(x)
        x = self.Agg(x, adj)
        x = self.ln(x)
        x = self.act(x)
        return x, adj

    def Agg(self, x, adj):
        b, n, _ = x.size()
        # b x n x 1 x d     0,0,...0,1,1,...1...
        x_left = torch.unsqueeze(x, 2)
        x_left = x_left.expand(-1, -1, n, -1)
        # b x 1 x n x d     0,1,...n-1,0,1,...n-1...
        x_right = torch.unsqueeze(x, 1)
        x_right = x_right.expand(-1, n, -1, -1)

        # --- 准备 msg_net 的输入 x_right_ ---
        if self.msg_transform:
            # 确保 adj_for_msg 与 x_right 的批处理维度兼容
            adj_for_msg = adj
            if adj.dim() == 2: # 假设 adj 是 (n, n)
                if b == 1:
                    adj_for_msg = adj.unsqueeze(0) # -> (1, n, n)
                else: # b > 1
                    adj_for_msg = adj.unsqueeze(0).expand(b, n, n) # -> (b, n, n)
            # adj_for_msg 现在应该是 (b, n, n)
            x_right_ = self.msg_net(torch.cat([x_right, adj_for_msg.unsqueeze(-1)], dim=-1)) # adj_for_msg.unsqueeze(-1) -> (b,n,n,1)
        else:
            # 修正：如果 msg_transform 为 False，通常 x_right_ 就是 x_right (或其简单变换)
            x_right_ = x_right # 直接使用 x_right 作为消息的基础

        # --- 准备 DenseAtt 的参数 ---
        # 1. adj_for_att (DenseAtt 的第三个参数)
        # 目标形状: (b, n, n, 1)
        processed_adj_for_att = adj
        if adj.dim() == 2: # 输入 adj 是 (n, n)
            if b == 1:
                processed_adj_for_att = adj.unsqueeze(0) # -> (1, n, n)
            else: # b > 1
                processed_adj_for_att = adj.unsqueeze(0).expand(b, n, n) # -> (b, n, n)
        # processed_adj_for_att 现在是 (b, n, n)
        adj_for_att = processed_adj_for_att.unsqueeze(-1) # -> (b, n, n, 1)

        # 2. edge_attr_for_att (DenseAtt 的第四个参数)
        edge_attr_for_att = None
        if self.edge_dim > 0: # 基于 GCLayer 初始化时的 self.edge_dim
            # 与 adj_for_att 使用相同的逻辑从原始 adj 生成
            # processed_adj_for_att 已经处理过批处理维度，直接使用
            edge_attr_for_att = processed_adj_for_att.unsqueeze(-1) # -> (b, n, n, 1)
        
        att = self.att(x_left, x_right, adj_for_att, edge_attr_for_att)
        msg = x_right_ * att
        msg = torch.sum(msg, dim=2)
        if self.sum_transform:
            msg = self.out_net(msg)
        x = x + msg
        return x


class GATLayer(nn.Module):


    def __init__(self, in_dim, out_dim, dropout=0., act=nn.LeakyReLU(0.5), edge_dim=0, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln',num_of_heads=4):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_proj = nn.Linear(in_dim, out_dim,bias=False)
        self.scoring_fn = nn.Linear(2*out_dim//num_of_heads+1,1,bias=False)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.num_of_heads = num_of_heads
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip_proj = None
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        x, adj = input

        b, n, _ = x.size()
        x = self.dropout(x)
        nodes_features_proj = self.linear_proj(x).view(b,n,self.num_of_heads,-1)  # (b,n,n_head,dim_out/n_head)
        nodes_features_proj = self.dropout(nodes_features_proj)
        x_left = torch.unsqueeze(nodes_features_proj, 2)
        x_left = x_left.expand(-1, -1, n, -1, -1)
        x_right = torch.unsqueeze(nodes_features_proj, 1)
        x_right = x_right.expand(-1, n, -1, -1, -1)  # (b,n,n,n_head,dim_out/n_head)
        score = self.scoring_fn(torch.cat([x_left,x_right,adj[...,None,None].expand(-1,-1,-1,self.num_of_heads,-1)],dim=-1)).squeeze()
        score = self.leakyReLU(score)  # (b,n,n,n_head)
        edge_mask = (adj > 1e-5).float()
        pad_mask = 1 - edge_mask
        zero_vec = -9e15 * pad_mask  # (b,n,n)

        att = score + zero_vec.unsqueeze(-1).expand(-1, -1,-1, self.num_of_heads)  # (b,n,n,n_head) padding的地方会-9e15
        att = torch.softmax(att,dim=2).transpose(2,3)  # (b,n,n_head,n)
        att = self.dropout(att).transpose(2, 3).unsqueeze(-1)
        msg = x_right * att  # (b,n,n,n_head,dim_out/n_head)
        msg = torch.sum(msg,dim=2)  # (b,n,n_head,dim_out/n_head)
        if self.in_dim != self.out_dim:
            x = self.skip_proj(x)  # (b,n,dim_out)

        x = x+msg.view(b,n,-1)+self.bias  # (b,n,dim_out)
        # x = self.ln(x)
        x = self.act(x)
        return x,adj

'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs
