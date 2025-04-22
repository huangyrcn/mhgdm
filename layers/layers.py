"""Euclidean layers."""
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from .att_layers import DenseAtt

import math
from typing import Any



def get_dim_act(config, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    model_config = config.model
    act = getattr(nn, model_config.act)
    acts = [act()] * (num_layers)

    if enc:
        dims = [model_config.hidden_dim] * (num_layers+1) # len=args.num_layers+1
    else:
        dims = [model_config.dim]+[model_config.hidden_dim] * (num_layers)   # len=args.num_layers+1

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
        self.att = DenseAtt(out_dim, dropout, edge_dim=edge_dim)
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

        if self.msg_transform:
            x_right_ = self.msg_net(torch.cat([x_right,adj.unsqueeze(-1)],dim=-1))
        else:
            x_right_ = x
        if self.edge_dim > 0:
            edge_attr = adj.unsqueeze(-1)
        else:
            edge_attr = None
        att = self.att(x_left, x_right,adj,edge_attr)  # (b*n_node*n_node,dim)
        msg = x_right_ * att
        msg = torch.sum(msg,dim=2)
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




def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

# -------- GCN layer --------
class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# -------- MLP layer --------
class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_classes: the number of classes of input, to be treated with different gains and biases,
                    (see the definition of class `ConditionalLayer1d`)
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


    def forward(self, x):
        """
        :param x: [num_classes * batch_size, N, F_i], batch of node features
            note that in self.cond_layers[layer],
            `x` is splited into `num_classes` groups in dim=0,
            and then treated with different gains and biases
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h)

