import numpy as np
import torch

import models.Decoders as Decoders

import models.Encoders as Encoders
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

    def __init__(self, device, encoder_class, encoder_params, decoder_class, decoder_params,
                 manifold_type, train_class_num, dim, pred_node_class=True, use_kl_loss=True,
                 use_base_proto_loss=True, use_sep_proto_loss=True, pred_edge=False,
                 pred_graph_class=False, classifier_dropout=0.0, classifier_bias=True):
        super(HVAE, self).__init__()
        self.device = device
        
        # Create encoder with specific parameters
        self.encoder = encoder_class(**encoder_params)

        # Create decoder with specific parameters
        if manifold_type != "Euclidean":
            # Check if decoder supports input_manifold parameter
            import inspect
            decoder_signature = inspect.signature(decoder_class.__init__)
            if 'input_manifold' in decoder_signature.parameters:
                decoder_params['input_manifold'] = self.encoder.manifolds[-1] if hasattr(self.encoder, 'manifolds') else None
        self.decoder = decoder_class(**decoder_params)

        self.loss_fn = OneHot_CrossEntropy()
        self.manifold = self.encoder.manifold

        # Store configuration parameters
        self.pred_node_class = pred_node_class
        self.use_kl_loss = use_kl_loss
        self.use_base_proto_loss = use_base_proto_loss
        self.use_sep_proto_loss = use_sep_proto_loss
        self.pred_edge_enabled = pred_edge
        self.pred_graph_class_enabled = pred_graph_class

        if self.pred_edge_enabled:
            self.edge_predictor = FermiDiracDecoder(self.encoder.manifold)
            self.edge_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.num_graph_classes = train_class_num
        self.latent_dim = dim
        # Revert prototype dimension to 2*latent_dim
        self.graph_prototypes = nn.Parameter(
            torch.randn(self.num_graph_classes, self.latent_dim * 2)
        )
        # Revert std calculation for 2*latent_dim
        std = 1.0 / ((self.latent_dim * 2) ** 0.5)
        nn.init.normal_(self.graph_prototypes, mean=0.0, std=std)
        self.current_epoch = 0

        # Initialize graph classification head if configured
        if self.pred_graph_class_enabled:
            # Use the Classifier from models.Decoders for graph-level classification
            self.graph_classifier = Decoders.Classifier(
                model_dim=dim,
                classifier_dropout=classifier_dropout,
                classifier_bias=classifier_bias,
                manifold=None,  # mean_graph is Euclidean
                n_classes=self.num_graph_classes,  # Graph-level classes
            )
            self.graph_classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.graph_classifier = None
            self.graph_classification_loss_fn = None

    def forward(self, x, adj, labels):  # labels are graph-level labels

        node_mask = node_flags(adj)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        node_mask = node_mask.unsqueeze(-1)

        posterior = self.encoder(x, adj, node_mask)
        h = posterior.sample()
        type_pred = self.decoder(h, adj, node_mask)

        if self.use_kl_loss:
            kl = posterior.kl()
        else:
            kl = torch.tensor(0.0, device=x.device)

        # Calculate node-level classification/reconstruction loss
        # Based on OneHot_CrossEntropy, this is effectively a node classification loss if x is one-hot
        if self.pred_node_class:
            node_classification_loss = self.loss_fn(type_pred * node_mask, x)
        else:
            node_classification_loss = torch.tensor(0.0, device=x.device)

        base_loss_proto = torch.tensor(0.0, device=x.device)
        loss_proto_separation = torch.tensor(0.0, device=x.device)
        mean_graph = None

        # Determine if we need to compute mean_graph (for base_proto_loss or graph_classification_loss)
        if self.use_base_proto_loss or self.pred_graph_class_enabled:
            emb_from_posterior = posterior.mode()  # Points on the manifold or in Euclidean space

            if emb_from_posterior.dim() == 2:
                emb_for_pooling = emb_from_posterior.unsqueeze(1)
            else:
                emb_for_pooling = emb_from_posterior

            if self.encoder.manifold is not None:
                emb_in_tangent_space = self.encoder.manifold.logmap0(emb_for_pooling)
            else:
                emb_in_tangent_space = emb_for_pooling

            mean_pooled_features = emb_in_tangent_space.mean(dim=1)
            max_pooled_features = emb_in_tangent_space.max(dim=1).values
            mean_graph = torch.cat([mean_pooled_features, max_pooled_features], dim=-1)

        if self.use_base_proto_loss:
            if mean_graph is None:
                # This should ideally not be reached if logic for mean_graph computation is correct
                raise ValueError(
                    "mean_graph is None but use_base_proto_loss is True. Check mean_graph computation logic."
                )
            current_graph_prototypes = self.graph_prototypes
            target_prototypes_for_batch = current_graph_prototypes[labels]
            distances_to_target_proto_sq = torch.sum(
                (mean_graph - target_prototypes_for_batch) ** 2, dim=-1
            )
            base_loss_proto = torch.mean(distances_to_target_proto_sq)

        if self.use_sep_proto_loss:
            current_graph_prototypes = self.graph_prototypes
            if current_graph_prototypes.shape[0] > 1:  # 确保有多于一个原型才进行计算
                # --- 新的计算方式：基于余弦相似度 ---
                # 1. 对原型进行 L2 归一化，使其成为单位向量，只关注方向
                prototypes_normalized = torch.nn.functional.normalize(
                    current_graph_prototypes, p=2, dim=-1, eps=1e-12
                )

                # 2. 计算归一化后原型之间的余弦相似度矩阵
                # (P_norm)^T * P_norm
                # 注意：这里应该是 prototypes_normalized 和其转置的乘积
                cosine_similarity_matrix = torch.matmul(
                    prototypes_normalized, prototypes_normalized.transpose(-2, -1)
                )

                # 3. 创建掩码，排除对角线元素（即一个原型与自身的相似度，恒为1）
                mask = ~torch.eye(
                    current_graph_prototypes.shape[0],  # 使用原型数量作为维度
                    dtype=torch.bool,
                    device=current_graph_prototypes.device,
                )

                # 4. 获取不同原型之间的余弦相似度值
                # cosine_similarity_matrix 的维度是 [num_prototypes, num_prototypes]
                # mask 的维度也应该是 [num_prototypes, num_prototypes]
                inter_proto_cosine_similarity = cosine_similarity_matrix[mask]

                # 5. 损失函数是这些余弦相似度的均值。
                # 目标是最小化这个值（即让它们方向尽可能不同，理想情况是负值或接近0）
                loss_proto_separation = inter_proto_cosine_similarity.mean()
                # --- 结束新的计算方式 ---
            else:
                # 如果只有一个原型或没有原型，则分离损失为0
                loss_proto_separation = torch.tensor(0.0, device=x.device)
        else:  # 如果不使用 sep_proto_loss
            loss_proto_separation = torch.tensor(0.0, device=x.device)

        # Graph classification loss calculation
        graph_classification_loss = torch.tensor(0.0, device=x.device)
        graph_classification_acc = 0.0

        if self.pred_graph_class_enabled:
            if mean_graph is None:
                # This should not happen if use_graph_classifier is true due to the combined check above
                # but as a safeguard or if logic changes:
                raise ValueError(
                    "mean_graph is None but pred_graph_class is True. This indicates a bug."
                )
            if self.graph_classifier is None or self.graph_classification_loss_fn is None:
                raise ValueError(
                    "Graph classifier or its loss function is not initialized. Check config for pred_graph_class."
                )

            # Get logits directly from the classifier's decode method
            # mean_graph is [Batch, DimLatent*2], suitable as 'h' for Classifier.decode
            # The 'adj' argument for Classifier.decode is not used by its internal cls layer.
            graph_class_logits = self.graph_classifier.decode(mean_graph, adj=None)

            graph_classification_loss = self.graph_classification_loss_fn(
                graph_class_logits, labels
            )
            pred_labels_graph_classifier = torch.argmax(graph_class_logits, dim=1)
            graph_classification_acc = (
                (pred_labels_graph_classifier == labels).float().mean().item()
            )

        if self.pred_edge_enabled:
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
            edge_loss = torch.tensor(0.0, device=x.device)  # Ensure float tensor for consistency

        # 只返回各项损失分量，不在模型内部合成总损失
        return (
            node_classification_loss / node_mask.sum(),
            kl.sum() / node_mask.sum(),
            edge_loss,
            base_loss_proto,
            loss_proto_separation,
            graph_classification_loss,  # Added graph classification loss
            graph_classification_acc,  # Added graph classification accuracy
        )


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
