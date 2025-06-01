import torch
import torch.nn as nn
from torch.nn import init

from layers.hyp_layers import HGCLayer, get_dim_act_curv, HGATLayer
from layers.euc_layers import GCLayer, get_dim_act, GATLayer
from layers.CentroidDistance import CentroidDistance
from utils import manifolds_utils

class Decoder(nn.Module):
    """ 
    Decoder abstract class
    """

    def __init__(self, max_feat_num, hidden_dim):
        super(Decoder, self).__init__()
        self.max_feat_num = max_feat_num
        self.hidden_dim = hidden_dim
        # self.expand_dim = nn.Linear(dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, max_feat_num),  # CrossEntropyLoss 内置了Softmax
        )

        # self.reset_parameters()
    #
    # def reset_parameters(self):
    #     init.xavier_uniform_(self.expand_dim.weight, gain=1)
    #     init.constant_(self.expand_dim.bias, 0)

    def forward(self, h, adj,node_mask):
        # h = self.expand_dim(h)
        output = self.decode(h,adj)
        type_pred = self.out(output)*node_mask
        return type_pred

class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, manifold):
        super(FermiDiracDecoder, self).__init__()
        self.manifold = manifold
        self.r = nn.Parameter(torch.ones((3,), dtype=torch.float))
        self.t = nn.Parameter(torch.ones((3,), dtype=torch.float))

    def forward(self, x):
        b, n, _ = x.size()
        x_left = x[:, :, None, :]
        x_right = x[:, None, :, :]
        if self.manifold is not None:
            dist = self.manifold.dist(x_left, x_right, keepdim=True)
        else:
            dist = torch.pairwise_distance(x_left, x_right, keepdim=True)  # (B,N,N,1)
        edge_type = 1.0 / (
            torch.exp((dist - self.r[None, None, None, :]) * self.t[None, None, None, :]) + 1.0
        )  # 对分子 改成3键 乘法变除法防止NaN
        noEdge = 1.0 - edge_type.max(dim=-1, keepdim=True)[0]
        edge_type = torch.cat([noEdge, edge_type], dim=-1)
        return edge_type


class Classifier(nn.Module):
    def __init__(
        self, model_dim, classifier_dropout, classifier_bias, manifold=None, n_classes=None
    ):
        super().__init__()
        self.manifold = manifold

        input_dim = model_dim * 2

        cls_layers = []
        if classifier_dropout > 0.0:
            cls_layers.append(nn.Dropout(p=classifier_dropout))
        cls_layers.append(nn.Linear(input_dim, n_classes, bias=classifier_bias))

        self.cls = nn.Sequential(*cls_layers)

    def decode(self, h, adj):
        """
        Processes input features through manifold mapping (if any) and self.cls.
        Returns raw class scores (logits).
        """
        h_processed = h
        if self.manifold is not None:
            # Use the manifold object's own logmap0 method
            h_mapped_to_tangent = self.manifold.logmap0(h)

            # Use the custom proj_tan0 function from manifolds_utils
            h_processed = manifolds_utils.proj_tan0(h_mapped_to_tangent, self.manifold)

        predictions = self.cls(h_processed)
        return predictions

    def forward(self, h, adj, node_mask):
        """
        Classifier's own forward pass.
        Calls decode to get final predictions and then applies node_mask.
        """
        predictions = self.decode(h, adj)
        return predictions * node_mask

class GCN(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, max_feat_num, hidden_dim, dec_layers, layer_type, dropout,
                 edge_dim, normalization_factor, aggregation_method, msg_transform):
        super(GCN, self).__init__(max_feat_num, hidden_dim)
        dims, acts = get_dim_act(
            hidden_dim=hidden_dim, 
            act_name='ReLU', 
            num_layers=dec_layers, 
            enc=False, 
            dim=max_feat_num
        )
        self.manifolds = None
        gc_layers = []
        if layer_type == 'GCN':
            layer_type_class = GCLayer
        elif layer_type == 'GAT':
            layer_type_class = GATLayer
        else:
            raise AttributeError(f"Unknown layer_type: {layer_type}")
        for i in range(dec_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(
                layer_type_class(
                    in_dim, out_dim, dropout, act, edge_dim,
                    normalization_factor, aggregation_method,
                    msg_transform
                )
            )
        self.layers = nn.Sequential(*gc_layers)
        self.message_passing = True

    def decode(self, x,adj):
        return self.layers((x,adj))[0]

class HGCN(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, max_feat_num, hidden_dim, dec_layers, layer_type, dropout,
                 edge_dim, normalization_factor, aggregation_method, msg_transform,
                 sum_transform, use_norm, manifold, c, learnable_c, use_centroid=False,
                 input_manifold=None):
        super(HGCN, self).__init__(max_feat_num, hidden_dim)
        dims, acts, self.manifolds = get_dim_act_curv(
            hidden_dim=hidden_dim,
            dim=max_feat_num,  # For decoder, this would be the output feature dimension
            manifold_name=manifold,
            c=c,
            learnable_c=learnable_c,
            act_name='ReLU',
            num_layers=dec_layers,
            enc=False
        )
        if input_manifold is not None:
            self.manifolds[0] = input_manifold
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if layer_type == 'HGCN':
            layer_type_class = HGCLayer
        elif layer_type == 'HGAT':
            layer_type_class = HGATLayer
        else:
            raise AttributeError(f"Unknown layer_type: {layer_type}")
        for i in range(dec_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                layer_type_class(
                    in_dim, out_dim, m_in, m_out, dropout, act, edge_dim,
                    normalization_factor, aggregation_method,
                    msg_transform, sum_transform, use_norm
                )
            )
        if use_centroid:
            self.centroid = CentroidDistance(dims[-1], dims[-1], self.manifold, dropout)
        self.layers = nn.Sequential(*hgc_layers)
        self.message_passing = True
        self.use_centroid = use_centroid

    def decode(self, x, adj):
        # x = proj_tan0(x, self.manifolds[0])
        # x = self.manifolds[0].expmap0(x)
        output,_ = self.layers((x,adj))
        if self.use_centroid:
            output = self.centroid(output)
        else:
            output = self.manifolds[-1].logmap0(output)
        return output

class CentroidDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, max_feat_num, hidden_dim, dim, manifold, dropout):
        super(CentroidDecoder, self).__init__(max_feat_num, hidden_dim)
        self.manifold = manifold
        self.centroid = CentroidDistance(hidden_dim, dim, self.manifold, dropout)
        self.message_passing = True

    def decode(self, x, adj):
        output = self.centroid(x)
        return output
def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u