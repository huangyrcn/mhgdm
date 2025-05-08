import torch
import torch.nn as nn
from torch.nn import init

from layers.hyp_layers import HGCLayer, get_dim_act_curv, HGATLayer
from layers.layers import GCLayer, get_dim_act, GATLayer
from layers.CentroidDistance import CentroidDistance
from torch.nn import functional as F


class Decoder(nn.Module):
    """
    Decoder abstract class
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.out = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.data.max_feat_num),
        )

    def forward(self, h, adj, node_mask):
        output = self.decode(h, adj)
        type_pred = self.out(output) * node_mask
        return type_pred


# Define a 3-layer classifier directly
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through each layer with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GCN(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, config):
        super(GCN, self).__init__(config)
        dims, acts = get_dim_act(config, config.model.dec_layers, enc=False)
        self.manifolds = None
        gc_layers = []
        if config.model.layer_type == "GCN":
            layer_type = GCLayer
        elif config.model.layer_type == "GAT":
            layer_type = GATLayer
        else:
            raise AttributeError
        for i in range(config.model.dec_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
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
        self.message_passing = True

    def decode(self, x, adj):
        return self.layers((x, adj))[0]


class HGCN(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, config, manifold=None):
        super(HGCN, self).__init__(config)
        dims, acts, self.manifolds = get_dim_act_curv(config, config.model.dec_layers, enc=False)
        if manifold is not None:
            self.manifolds[0] = manifold
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if config.model.layer_type == "HGCN":
            layer_type = HGCLayer
        elif config.model.layer_type == "HGAT":
            layer_type = HGATLayer
        else:
            raise AttributeError
        for i in range(config.model.dec_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
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
        if config.model.use_centroid:
            self.centroid = CentroidDistance(
                dims[-1], dims[-1], self.manifold, config.model.dropout
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.message_passing = True

    def decode(self, x, adj):
        output, _ = self.layers((x, adj))
        if self.config.model.use_centroid:
            output = self.centroid(output)
        else:
            output = self.manifolds[-1].logmap0(output)
        return output


class CentroidDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, config, manifold=None):
        super(CentroidDecoder, self).__init__(config)
        self.manifold = manifold
        self.centroid = CentroidDistance(
            config.model.hidden_dim, config.model.dim, self.manifold, config.model.dropout
        )
        self.message_passing = True

    def decode(self, x, adj):
        output = self.centroid(x)
        return output


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.bn = nn.BatchNorm1d(ft_in)

    def forward(self, seq):
        ret = self.fc(self.bn(seq))
        return ret
