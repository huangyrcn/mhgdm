import numpy as np
import torch

import layers.Decoders as Decoders

import layers.Encoders as Encoders
from torch import nn
from utils.graph_utils import node_flags


class OneHot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(OneHot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=2)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.sum(torch.sum(loss, dim=2))
        return loss


class HVAE(nn.Module):

    def __init__(self, config):
        super(HVAE, self).__init__()
        self.device = config.device
        self.encoder = getattr(Encoders, config.model.encoder)(config)

        if config.model.manifold != "Euclidean":
            self.decoder = getattr(Decoders, config.model.decoder)(
                config, self.encoder.manifolds[-1]
            )
        else:
            self.decoder = getattr(Decoders, config.model.decoder)(config)
        self.config = config

        self.loss_fn = OneHot_CrossEntropy()
        self.manifold = self.encoder.manifold

        if config.model.pred_edge:
            self.edge_predictor = FermiDiracDecoder(self.encoder.manifold)
            self.edge_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.num_graph_classes = config.data.train_class_num  # 假设你在 config 里有 num_classes
        self.latent_dim = config.model.dim
        self.graph_prototypes = nn.Parameter(
            torch.randn(self.num_graph_classes, self.latent_dim * 2)
        )
        # 合理初始化图类别原型参数
        std = 1.0 / ((self.latent_dim * 2) ** 0.5)
        nn.init.normal_(self.graph_prototypes, mean=0.0, std=std)

    def forward(self, x, adj, labels):

        node_mask = node_flags(adj)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        node_mask = node_mask.unsqueeze(-1)

        posterior = self.encoder(x, adj, node_mask)
        h = posterior.sample()
        type_pred = self.decoder(h, adj, node_mask)

        kl = posterior.kl()

        loss = self.loss_fn(type_pred * node_mask, x)
        loss_proto = torch.tensor(0, device=x.device) 
        if self.config.model.use_proto_loss:
            mean = posterior.mode()
            if mean.dim() == 3 and mean.size(1) == 1:
                mean = mean.squeeze(1)
            mean_mean = mean.mean(dim=1)
            mean_max = mean.max(dim=1).values
            mean_graph = torch.cat([mean_mean, mean_max], dim=-1)  # 拼接平均池化和最大池化
            target_prototypes = self.graph_prototypes[labels]  # (batch_size, latent_dim*2)
            distances_to_proto = self.manifold.dist(mean_graph, target_prototypes)  # (batch_size,)
            loss_proto = torch.mean(distances_to_proto**2)

        if self.config.model.pred_edge:
            triu_mask = torch.triu(edge_mask, 1)[:, :, :, None]
            edge_pred = self.edge_predictor(posterior.mode()) * triu_mask
            edge_pred = edge_pred.view(-1, 4)  # 4 for edge type
            triu_mask = triu_mask.view(-1).cpu().numpy().astype(np.bool)
            adj = torch.triu(adj, 1).long().view(-1)

            adj_numpy = adj.cpu().numpy().astype(np.bool)
            adj_numpy_invert = ~adj_numpy * triu_mask  # 既不是正边也不是pad
            pos_edges = np.where(adj_numpy)[0]
            neg_edges = np.where(adj_numpy_invert)[0]
            choise_num = np.min([len(pos_edges), len(neg_edges)])
            pos_id = np.random.choice(len(pos_edges), choise_num)
            neg_id = np.random.choice(len(neg_edges), choise_num)
            pos_id = pos_edges[pos_id]
            neg_id = neg_edges[neg_id]
            choose_id = torch.tensor(np.append(pos_id, neg_id))
            edge_loss = self.edge_loss_fn(edge_pred[choose_id], adj[choose_id])
        else:
            edge_loss = torch.tensor(0, device=x.device)

        # 只返回各项损失分量，不在模型内部合成总损失
        return loss / node_mask.sum(), kl.sum() / node_mask.sum(), edge_loss, loss_proto

    def show_curvatures(self):
        if self.config.model.manifold != "Euclidean":
            c = [f"{m.k.item():.4f}" for m in self.encoder.manifolds]
            if hasattr(self.decoder, "manifolds"):
                c.append([f"{m.k.item():.4f}" for m in self.decoder.manifolds])
            print(c)


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
