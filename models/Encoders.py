import torch
from geoopt import ManifoldParameter
from torch import nn
from utils.Distributions import DiagonalGaussianDistribution
from layers.hyp_layers import get_dim_act_curv, HGCLayer, HGATLayer
from layers.euc_layers import get_dim_act, GCLayer, GATLayer


def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    return torch.sqrt(radial + 1e-8)


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, input_feat_dim, hidden_dim, dim):
        super(Encoder, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.embedding = nn.Linear(self.input_feat_dim, self.hidden_dim, bias=False)
        self.mean_logvar_net = nn.Linear(self.hidden_dim, 2 * self.dim)

    def forward(self, x, adj, node_mask):
        x = self.embedding(x)  # (b,n_atom,n_atom_embed)
        output = self.encode(x, adj)
        mean_logvar = self.mean_logvar_net(output)
        posterior = DiagonalGaussianDistribution(mean_logvar, self.manifold, node_mask)

        return posterior


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, input_feat_dim, hidden_dim, dim, enc_layers, layer_type='GCN', 
                 dropout=0.0, edge_dim=1, normalization_factor=1.0, 
                 aggregation_method='sum', msg_transform='linear'):
        super(GCN, self).__init__(input_feat_dim, hidden_dim, dim)
        self.enc_layers = enc_layers
        
        # 获取维度和激活函数
        self.dims, self.acts = get_dim_act(
            hidden_dim=self.hidden_dim, 
            act_name='ReLU',  # 默认激活函数
            num_layers=self.enc_layers
        )
        self.manifold = None
        gc_layers = []
        if layer_type == 'GCN':
            layer_type_class = GCLayer
        elif layer_type == 'GAT':
            layer_type_class = GATLayer
        else:
            raise AttributeError(f"Unknown layer_type: {layer_type}")
        for i in range(self.enc_layers):
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            gc_layers.append(
                layer_type_class(
                    in_dim, out_dim, dropout, act, edge_dim,
                    normalization_factor, aggregation_method,
                    msg_transform, msg_transform  # sum_transform用msg_transform替代
                )
            )
        self.layers = nn.Sequential(*gc_layers)

    def encode(self, x, adj):
        output, _ = self.layers((x, adj))
        return output


class HGCN(Encoder):
    """
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, input_feat_dim, hidden_dim, dim, enc_layers, layer_type='HGCN',
                 dropout=0.0, edge_dim=1, normalization_factor=1.0, 
                 aggregation_method='sum', msg_transform='linear', sum_transform='linear',
                 use_norm=False, manifold='PoincareBall', c=1.0, learnable_c=False):
        super(HGCN, self).__init__(input_feat_dim, hidden_dim, dim)
        self.enc_layers = enc_layers
        self.dims, self.acts, self.manifolds = get_dim_act_curv(
            hidden_dim=self.hidden_dim,
            dim=self.dim, 
            manifold_name=manifold,
            c=c,
            learnable_c=learnable_c,
            act_name='ReLU',
            num_layers=self.enc_layers,
            enc=True
        )
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if layer_type == 'HGCN':
            layer_type_class = HGCLayer
        elif layer_type == 'HGAT':
            layer_type_class = HGATLayer
        else:
            raise AttributeError(f"Unknown layer_type: {layer_type}")
        for i in range(self.enc_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            hgc_layers.append(
                layer_type_class(
                    in_dim, out_dim, m_in, m_out, dropout, act, edge_dim,
                    normalization_factor, aggregation_method,
                    msg_transform, sum_transform, use_norm
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        std = 1 / torch.sqrt(torch.abs(self.manifold.k))
        self.embedding.weight = ManifoldParameter(
            self.manifolds[0].random_normal((self.input_feat_dim, self.hidden_dim), std=std).T, self.manifolds[0]
        )

    def encode(self, x, adj):

        output, _ = self.layers((x, adj))

        output = self.manifolds[-1].logmap0(output)

        return output

    def proj_tan0(self, u, manifold):
        if manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u
