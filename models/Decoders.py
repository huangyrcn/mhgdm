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

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        # self.expand_dim = nn.Linear(config.model.dim, config.model.hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.data.max_feat_num),  # CrossEntropyLoss 内置了Softmax
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


class Classifier(nn.Module):
    def __init__(
        self, model_dim, classifier_dropout, classifier_bias, manifold=None, n_classes=None
    ):
        super().__init__()
        self.manifold = manifold

        input_dim = model_dim * 2

        final_output_dim = n_classes

        classifier_dropout_rate = classifier_dropout
        classifier_use_bias = classifier_bias

        cls_layers = []
        if classifier_dropout_rate > 0.0:
            cls_layers.append(nn.Dropout(p=classifier_dropout_rate))
        cls_layers.append(nn.Linear(input_dim, final_output_dim, bias=classifier_use_bias))

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

    def __init__(self, config):
        super(GCN, self).__init__(config)
        dims, acts = get_dim_act(config, config.model.dec_layers, enc=False)
        self.manifolds = None
        gc_layers = []
        if config.model.layer_type == 'GCN':
            layer_type = GCLayer
        elif config.model.layer_type == 'GAT':
            layer_type = GATLayer
        else:
            raise AttributeError
        for i in range(config.model.dec_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(
                layer_type(
                    in_dim, out_dim, config.model.dropout, act, config.model.edge_dim,
                    config.model.normalization_factor,config.model.aggregation_method,
                    config.model.aggregation_method, config.model.msg_transform
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

    def __init__(self, config, manifold=None):
        super(HGCN, self).__init__(config)
        dims, acts, self.manifolds = get_dim_act_curv(config, config.model.dec_layers, enc=False)
        if manifold is not None:
            self.manifolds[0] = manifold
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if config.model.layer_type == 'HGCN':
            layer_type = HGCLayer
        elif config.model.layer_type == 'HGAT':
            layer_type = HGATLayer
        else:
            raise AttributeError
        for i in range(config.model.dec_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                layer_type(
                    in_dim, out_dim, m_in, m_out, config.model.dropout, act, config.model.edge_dim,
                    config.model.normalization_factor,config.model.aggregation_method,
                    config.model.msg_transform, config.model.sum_transform, config.model.use_norm
                )
            )
        if config.model.use_centroid:
            self.centroid = CentroidDistance(dims[-1], dims[-1], self.manifold, config.model.dropout)
        self.layers = nn.Sequential(*hgc_layers)
        self.message_passing = True

    def decode(self, x, adj):
        # x = proj_tan0(x, self.manifolds[0])
        # x = self.manifolds[0].expmap0(x)
        output,_ = self.layers((x,adj))
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
        self.centroid = CentroidDistance(config.model.hidden_dim, config.model.dim, self.manifold, config.model.dropout)
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