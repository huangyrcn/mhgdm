"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from geoopt import PoincareBall
from geoopt import Lorentz
# from manifolds import Lorentz
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt
from utils.manifolds_utils import proj_tan0, proj_tan, exp_after_transp0, transp0back_after_logmap


def get_dim_act_curv(hidden_dim, dim, manifold_name='PoincareBall', c=1.0, 
                     learnable_c=False, act_name='ReLU', num_layers=3, enc=True):
    """
    Helper function to get dimension, activation, and manifolds at every layer for hyperbolic models.
    
    Args:
        hidden_dim: Hidden layer dimension
        dim: Latent space dimension (for decoders)
        manifold_name: Name of the manifold ('PoincareBall' or 'Lorentz')
        c: Curvature value
        learnable_c: Whether curvature is learnable
        act_name: Activation function name (e.g., 'ReLU', 'LeakyReLU')
        num_layers: Number of layers
        enc: Whether this is for encoder (True) or decoder (False)
        
    Returns:
        dims: List of layer dimensions
        acts: List of activation functions
        manifolds: List of manifold instances
    """
    act_class = getattr(nn, act_name)
    if isinstance(act_class(), nn.LeakyReLU):
        acts = [act_class(0.5)] * num_layers
    else:
        acts = [act_class()] * num_layers

    if enc:
        dims = [hidden_dim] * (num_layers + 1)  # len=num_layers+1
    else:
        dims = [dim] + [hidden_dim] * num_layers  # len=num_layers+1

    manifold_class = {'PoincareBall': PoincareBall, 'Lorentz': Lorentz}

    if enc:
        manifolds = [manifold_class[manifold_name](c, learnable=learnable_c)
                     for _ in range(num_layers)] + [manifold_class[manifold_name](c, learnable=learnable_c)]
    else:
        manifolds = [manifold_class[manifold_name](c, learnable=learnable_c)] + \
                    [manifold_class[manifold_name](c, learnable=learnable_c) for _ in range(num_layers)]

    return dims, acts, manifolds


# Backward compatibility function
def get_dim_act_curv_legacy(config, num_layers, enc=True):
    """
    Backward compatibility wrapper for get_dim_act_curv.
    Uses config object to extract parameters.
    """
    model_config = config.model
    return get_dim_act_curv(
        hidden_dim=model_config.hidden_dim,
        dim=model_config.dim,
        manifold_name=model_config.manifold,
        c=model_config.c,
        learnable_c=model_config.learnable_c,
        act_name=model_config.act,
        num_layers=num_layers,
        enc=enc
    )


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm=True):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in, dropout)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)
        self.norm = HypNorm(out_dim, manifold_in)
        self.use_norm = use_norm
    def forward(self, x):
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.hyp_act(x)
        return x


class HyperbolicGraphConvolutionLayer(nn.Module):
    """
    Hyperbolic Graph Convolution Layer.
    Improved version with clearer parameter naming and documentation.
    """

    def __init__(self, input_feature_dim, output_feature_dim, input_manifold, output_manifold, 
                 dropout=0.0, activation=nn.ReLU(), edge_feature_dim=1, normalization_factor=1.0,
                 aggregation_method='sum', use_message_transform=True, 
                 use_output_transform=True, normalization_type='ln'):
        super(HyperbolicGraphConvolutionLayer, self).__init__()
        self.linear = HypLinear(input_feature_dim, output_feature_dim, input_manifold, dropout)
        self.aggregation = HypAgg(
            output_feature_dim, input_manifold, dropout, edge_feature_dim, normalization_factor, 
            aggregation_method, activation, use_message_transform, use_output_transform
        )
        self.normalization_type = normalization_type
        if normalization_type != 'none':
            self.normalization = HypNorm(output_feature_dim, input_manifold, normalization_type)
        self.hyperbolic_activation = HypAct(input_manifold, output_manifold, activation)

    def forward(self, input_data):
        node_features, adjacency_matrix = input_data
        node_features = self.linear(node_features)
        node_features = self.aggregation(node_features, adjacency_matrix)
        if self.normalization_type != 'none':
            node_features = self.normalization(node_features)         
        node_features = self.hyperbolic_activation(node_features)
        return node_features, adjacency_matrix


# Backward compatibility alias
class HGCLayer(HyperbolicGraphConvolutionLayer):
    """
    Backward compatibility alias for HyperbolicGraphConvolutionLayer.
    Maps old parameter names to new ones.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), 
                 edge_dim=1, normalization_factor=1, aggregation_method='sum', 
                 msg_transform=True, sum_transform=True, use_norm='ln'):
        super(HGCLayer, self).__init__(
            input_feature_dim=in_dim,
            output_feature_dim=out_dim,
            input_manifold=manifold_in,
            output_manifold=manifold_out,
            dropout=dropout,
            activation=act,
            edge_feature_dim=edge_dim,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            use_message_transform=msg_transform,
            use_output_transform=sum_transform,
            normalization_type=use_norm
        )


class HGATLayer(nn.Module):

    """https://github.com/gordicaleksa/pytorch-GAT"""
    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.LeakyReLU(0.5), edge_dim=2, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln',num_of_heads=4,local_agg=True,use_act=True,
                 return_multihead=False):
        super(HGATLayer, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.linear_proj = nn.Linear(in_dim, out_dim,bias=False)
        self.scoring_fn = nn.Linear(2*out_dim//num_of_heads+1,1,bias=False)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.use_act = use_act
        self.act = act
        self.num_of_heads = num_of_heads
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip_proj = None
        self.local_agg = local_agg
        self.return_multihead = return_multihead
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        x, adj = input
        b, n, _ = x.size()
        x = self.manifold_in.logmap0(x)  # (b,n,dim_in)
        x = self.dropout(x)
        nodes_features_proj = self.linear_proj(x).view(b,n,self.num_of_heads,-1)  # (b,n,n_head,dim_out/n_head)
        nodes_features_proj = self.dropout(nodes_features_proj)
        # score_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)  # (b,n,n_head)
        # score_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)  # (b,n,n_head)
        # score = self.leakyReLU(score_source.unsqueeze(1) + score_target.unsqueeze(2))
        x_left = torch.unsqueeze(nodes_features_proj, 2)
        x_left = x_left.expand(-1, -1, n, -1, -1)
        x_right = torch.unsqueeze(nodes_features_proj, 1)
        x_right = x_right.expand(-1, n, -1, -1, -1)  # (b,n,n,n_head,dim_out/n_head)
        score = self.scoring_fn(torch.cat([x_left,x_right,adj[...,None,None].expand(-1,-1,-1,self.num_of_heads,-1)],dim=-1)).squeeze()
        score = self.leakyReLU(score)  # (b,n,n,n_head)

        if self.local_agg:
            edge_mask = (adj > 1e-5).float()
            pad_mask = 1 - edge_mask
            connectivity_mask = -9e15 * pad_mask  # (b,n,n)
            score = score + connectivity_mask.unsqueeze(-1).expand(-1, -1,-1, self.num_of_heads)  # (b,n,n,n_head) padding的地方会-9e15

        att = torch.softmax(score,dim=2).transpose(2,3)  # (b,n,n_head,n)
        att = self.dropout(att).transpose(2,3).unsqueeze(-1)  # (b,n,n,n_head,1)
        msg = x_right * att  # (b,n,n,n_head,dim_out/n_head)
        msg = torch.sum(msg,dim=2)  # ->(b,n,n_head,dim_out/n_head)
        if self.return_multihead:
            return self.manifold_out.expmap0(msg)
        if self.in_dim != self.out_dim:
            x = self.skip_proj(x)  # (b,n,dim_out)
        x = self.manifold_out.expmap0(x)
        addend = msg.view(b,n,-1)+self.bias
        addend = self.manifold_out.transp0(x,addend)
        x = self.manifold_out.expmap(x,addend)
        if self.use_act:
            x = self.act(x)     # Todo 是否需要 可能encoder会有问题
        return x,adj


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, manifold_in, dropout):
        super(HypLinear, self).__init__()
        self.manifold = manifold_in
        self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dp = nn.Dropout(dropout)
        if out_dim > in_dim and self.manifold.name == 'Lorentz':
            self.scale = nn.Parameter(torch.tensor([1 / out_dim]).sqrt_())
        else:
            self.scale = 1.
        self.reset_parameters()

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x) * self.scale
        x = self.dp(x)
        x = proj_tan0(x, self.manifold)
        x = self.manifold.expmap0(x)
        bias = proj_tan0(self.bias.view(1, -1), self.manifold)
        bias = self.manifold.transp0(x, bias)
        x = self.manifold.expmap(x, bias)
        return x


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, out_dim, manifold_in, dropout, edge_dim, normalization_factor=1, aggregation_method='sum',
                 act=nn.ReLU(), msg_transform=True, sum_transform=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold_in
        self.dim = out_dim
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.att_net = DenseAtt(out_dim, dropout=dropout, edge_dim=edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        if msg_transform:
            self.msg_net = nn.Sequential(
                nn.Linear(out_dim+edge_dim-1, out_dim),
                act,
                # nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )

        if sum_transform:
            self.out_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                # nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )


    def forward(self, x, adj):
        b,n,_ = x.size()
        # # b x n x 1 x d     0,0,...0,1,1,...1...
        # x_left = torch.unsqueeze(x, 2)
        # x_left = x_left.expand(-1,-1, n, -1)
        # # b x 1 x n x d     0,1,...n-1,0,1,...n-1...
        # x_right = torch.unsqueeze(x, 1)
        # x_right = x_right.expand(-1,n, -1, -1)
        x_left = x[:,:, None,:].expand(-1,-1, n, -1)
        x_right = x[:,None, :,:].expand(-1,n, -1, -1)
        edge_attr = None
        if self.edge_dim == 1:
            edge_attr = None
        elif self.edge_dim == 2:
            edge_attr = self.manifold.dist(x_left, x_right, keepdim=True)  # (b,n,n,1)

        att = self.att_net(self.manifold.logmap0(x_left), self.manifold.logmap0(x_right),adj.unsqueeze(-1),edge_attr)  # (b,n_node,n_node,dim)
        if self.msg_transform:
            msg = self.manifold.logmap0(x_right)
            if edge_attr is not None:
                msg = torch.cat([msg,edge_attr],dim=-1)
            msg = self.msg_net(msg)
        else:
            msg = self.manifold.logmap(x_left, x_right)# (b,n_node,n_node,dim)  x_col落在x_row的切空间
        msg = msg * att
        msg = torch.sum(msg,dim=2)  # (b,n_node,dim)
        if self.sum_transform:
            msg = self.out_net(msg)
        if self.msg_transform:
            msg = proj_tan0(msg, self.manifold)
            msg = self.manifold.transp0(x, msg)
        else:
            msg = proj_tan(x, msg,self.manifold)
        output = self.manifold.expmap(x, msg)
        return output


class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        x = self.act(self.manifold_in.logmap0(x))
        x = proj_tan0(x, self.manifold_in)
        x = self.manifold_out.expmap0(x)
        return x


class HypNorm(nn.Module):

    def __init__(self, in_features, manifold, method='ln'):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        if self.manifold.name == 'Lorentz':
            in_features = in_features - 1
        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)

    def forward(self, h):
        h = self.manifold.logmap0(h)
        if self.manifold.name == 'Lorentz':
            h[..., 1:] = self.norm(h[..., 1:].clone())
        else:
            h = self.norm(h)
        h = self.manifold.expmap0(h)
        return h

    
def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u


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
