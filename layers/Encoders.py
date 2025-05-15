from sklearn import manifold
import torch
from geoopt import ManifoldParameter
from torch import nn
from utils.Distributions import DiagonalGaussianDistribution
from layers.hyp_layers import get_dim_act_curv, HGCLayer, HGATLayer
from layers.layers import get_dim_act, GCLayer, GATLayer
import geoopt
from utils.manifolds_utils import is_on_lorentz_manifold


def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    return torch.sqrt(radial + 1e-8)


def initialization_on_manifold(x_euclidean: torch.Tensor, manifold: geoopt.Manifold):
    if manifold is None:
        # 欧氏空间，直接返回
        return x_euclidean
    if isinstance(manifold, geoopt.Lorentz):
        # 限制范数，避免过大带来数值不稳定
        x_euclidean = torch.nn.functional.normalize(x_euclidean, dim=-1) * 0.1
        # 拼接时间维度为 0 的切向量
        zero_time = torch.zeros_like(x_euclidean[..., :1])
        x_tangent = torch.cat([zero_time, x_euclidean], dim=-1)  # (..., d+1)
        x_hyperbolic = manifold.expmap0(x_tangent)
        # 确保点严格位于双曲流形上
        x_hyperbolic = manifold.projx(x_hyperbolic)
        return x_hyperbolic

    elif isinstance(manifold, geoopt.PoincareBall):
        # 控制模长避免靠近边界
        x_euclidean = torch.tanh(x_euclidean) * 0.9
        x_hyperbolic = manifold.expmap0(x_euclidean)
        # 确保点严格位于双曲流形上
        x_hyperbolic = manifold.projx(x_hyperbolic)
        return x_hyperbolic

    else:
        raise NotImplementedError(f"Unsupported manifold type: {type(manifold)}")


class Encoder(nn.Module):
    """
    Encoder abstract class. encoder 最终返回一个 流形上的分布
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        if self.config.model.manifold == "Lorentz":
            self.embedding = nn.Linear(
                config.data.max_feat_num, config.model.hidden_dim - 1, bias=False
            )
        else:
            self.embedding = nn.Linear(
                config.data.max_feat_num, config.model.hidden_dim, bias=False
            )
        self.mean_logvar_net = nn.Linear(config.model.hidden_dim, 2 * config.model.dim)

    def forward(self, x, adj, node_mask):

        x = self.embedding(x)
        x = initialization_on_manifold(x, self.manifold)
        output = self.encode(x, adj)
        mean_logvar = self.mean_logvar_net(output)
        posterior = DiagonalGaussianDistribution(mean_logvar, self.manifold, node_mask)
        return posterior


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, config):
        super(GCN, self).__init__(config)
        self.dims, self.acts = get_dim_act(config, config.model.enc_layers)
        self.manifold = None
        gc_layers = []
        if config.model.layer_type == "GCN":
            layer_type = GCLayer
        elif config.model.layer_type == "GAT":
            layer_type = GATLayer
        else:
            raise AttributeError
        for i in range(config.model.enc_layers):
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            gc_layers.append(
                layer_type(
                    in_dim,
                    out_dim,
                    config.model.dropout,
                    act,
                    config.model.edge_dim,
                    config.model.normalization_factor,
                    config.model.aggregation_method,
                    config.model.aggregation_method,
                    config.model.msg_transform,
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

    def __init__(self, config):
        super(HGCN, self).__init__(config)
        self.dims, self.acts, self.manifolds = get_dim_act_curv(config, config.model.enc_layers)
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if config.model.layer_type == "HGCN":
            layer_type = HGCLayer
        elif config.model.layer_type == "HGAT":
            layer_type = HGATLayer
        else:
            raise AttributeError
        for i in range(config.model.enc_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            hgc_layers.append(
                layer_type(
                    in_dim,
                    out_dim,
                    m_in,
                    m_out,
                    config.model.dropout,
                    act,
                    config.model.edge_dim,
                    config.model.normalization_factor,
                    config.model.aggregation_method,
                    config.model.msg_transform,
                    config.model.sum_transform,
                    config.model.use_norm,
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        std = 1 / torch.sqrt(torch.abs(self.manifold.k))
        self.embedding.weight = ManifoldParameter(
            self.manifolds[0]
            .random_normal((self.embedding.in_features, self.embedding.out_features), std=std)
            .T,
            self.manifolds[0],
        )

    def encode(self, x, adj):

        output, _ = self.layers((x, adj))

        output = self.manifolds[-1].logmap0(output)

        return output

    def proj_tan0(self, u, manifold):
        if manifold.name == "Lorentz":
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u
