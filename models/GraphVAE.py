import numpy as np
import torch

import models.Decoders as Decoders

import models.Encoders as Encoders
from torch import nn
from utils.graph_utils import node_flags


class OneHotCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss for one-hot encoded targets."""

    def __init__(self):
        super(OneHotCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss between predictions and one-hot targets.

        Args:
            predictions: Model predictions [batch_size, num_nodes, num_classes]
            targets: One-hot encoded targets [batch_size, num_nodes, num_classes]
        """
        probabilities = torch.nn.functional.softmax(predictions, dim=2)
        log_prob_loss = targets * torch.log(
            probabilities + 1e-7
        )  # Add small epsilon for numerical stability
        total_loss = -torch.sum(torch.sum(log_prob_loss, dim=2))
        return total_loss


class GraphVAE(nn.Module):

    def __init__(self, config):
        """
        Initialize GraphVAE with a single configuration object.

        Args:
            config: Configuration object containing all model parameters including:
                - pred_node_class: Enable node-level prediction/reconstruction
                - pred_edge: Enable edge prediction
                - use_kl_loss: Enable KL divergence loss
                - use_base_proto_loss: Enable base prototype loss
                - use_sep_proto_loss: Enable prototype separation loss
                - encoder_config: Configuration for encoder (must have 'type' attribute)
                - decoder_config: Configuration for decoder (must have 'type' attribute)
                - edge_predictor_config: Configuration for edge predictor
                - latent_dim: Latent dimension
                - device: Device for computation
        """
        super(GraphVAE, self).__init__()

        # Validate required configs
        encoder_config = getattr(config, "encoder_config", None)
        decoder_config = getattr(config, "decoder_config", None)

        if encoder_config is None:
            raise ValueError("encoder_config is required")
        if decoder_config is None:
            raise ValueError("decoder_config is required")

        # Extract parameters from config with defaults
        self.device = getattr(config, "device", "cpu")
        self.pred_node_class = getattr(config, "pred_node_class", True)
        self.pred_edge_enabled = getattr(config, "pred_edge", False)
        self.use_kl_loss = getattr(config, "use_kl_loss", True)
        self.use_base_proto_loss = getattr(config, "use_base_proto_loss", False)
        self.use_sep_proto_loss = getattr(config, "use_sep_proto_loss", False)
        self.latent_dim = getattr(config, "latent_dim", 10)

        # Extract sub-module configurations
        edge_predictor_config = getattr(config, "edge_predictor_config", None)

        # Initialize encoder and decoder with their specific configs
        self._initialize_encoder_decoder(encoder_config, decoder_config)

        # Loss function for node classification/reconstruction
        self.node_reconstruction_loss_fn = OneHotCrossEntropyLoss()
        # Alias for backward compatibility
        self.loss_fn = self.node_reconstruction_loss_fn

        # Get manifold from encoder
        self.manifold = getattr(self.encoder, "manifold", None)

        # Initialize edge prediction if enabled
        if self.pred_edge_enabled:
            self._initialize_edge_prediction(edge_predictor_config)

        # Initialize graph prototypes for prototype loss if enabled
        if self.use_base_proto_loss or self.use_sep_proto_loss:
            self._initialize_graph_prototypes()

        self.current_epoch = 0

    def _initialize_encoder_decoder(self, encoder_config, decoder_config):
        """Initialize encoder and decoder based on their specific configurations."""
        # Initialize encoder
        if not hasattr(encoder_config, "type"):
            raise ValueError("encoder_config must have 'type' attribute")

        encoder_type = encoder_config.type
        if encoder_type == "EuclideanGraphEncoder":
            self.encoder = Encoders.EuclideanGraphEncoder(
                input_feature_dim=encoder_config.input_feature_dim,
                hidden_feature_dim=getattr(encoder_config, "hidden_feature_dim", 32),
                latent_feature_dim=getattr(encoder_config, "latent_feature_dim", 10),
                num_encoder_layers=getattr(encoder_config, "num_layers", 3),
                layer_type=getattr(encoder_config, "layer_type", "GCN"),
                dropout=getattr(encoder_config, "dropout", 0.0),
                edge_dim=getattr(encoder_config, "edge_dim", 1),
                normalization_factor=getattr(encoder_config, "normalization_factor", 1.0),
                aggregation_method=getattr(encoder_config, "aggregation_method", "sum"),
                message_transformation=getattr(encoder_config, "message_transformation", "linear"),
            )
        elif encoder_type == "HyperbolicGraphEncoder":
            self.encoder = Encoders.HyperbolicGraphEncoder(
                input_feature_dim=getattr(encoder_config, "input_feature_dim", 3),
                hidden_feature_dim=getattr(encoder_config, "hidden_feature_dim", 32),
                latent_feature_dim=getattr(encoder_config, "latent_feature_dim", 10),
                num_encoder_layers=getattr(encoder_config, "num_layers", 3),
                layer_type=getattr(encoder_config, "layer_type", "HGAT"),
                dropout=getattr(encoder_config, "dropout", 0.0),
                edge_dim=getattr(encoder_config, "edge_dim", 1),
                normalization_factor=getattr(encoder_config, "normalization_factor", 1.0),
                aggregation_method=getattr(encoder_config, "aggregation_method", "sum"),
                message_transformation=getattr(encoder_config, "message_transformation", "linear"),
                aggregation_transformation=getattr(
                    encoder_config, "aggregation_transformation", "linear"
                ),
                use_normalization=getattr(encoder_config, "use_normalization", False),
                manifold_type=getattr(encoder_config, "manifold_type", "PoincareBall"),
                curvature=getattr(encoder_config, "curvature", 1.0),
                learnable_curvature=getattr(encoder_config, "learnable_curvature", False),
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Initialize decoder
        if not hasattr(decoder_config, "type"):
            raise ValueError("decoder_config must have 'type' attribute")

        decoder_type = decoder_config.type
        if decoder_type == "EuclideanGraphDecoder":
            self.decoder = Decoders.EuclideanGraphDecoder(
                latent_feature_dim=getattr(decoder_config, "latent_feature_dim", 10),
                hidden_feature_dim=getattr(decoder_config, "hidden_feature_dim", 32),
                output_feature_dim=getattr(decoder_config, "output_feature_dim", 3),
                num_decoder_layers=getattr(decoder_config, "num_layers", 3),
                layer_type=getattr(decoder_config, "layer_type", "GCN"),
                dropout=getattr(decoder_config, "dropout", 0.0),
                edge_dim=getattr(decoder_config, "edge_dim", 1),
                normalization_factor=getattr(decoder_config, "normalization_factor", 1.0),
                aggregation_method=getattr(decoder_config, "aggregation_method", "sum"),
                message_transformation=getattr(decoder_config, "message_transformation", "linear"),
            )
        elif decoder_type == "HyperbolicGraphDecoder":
            self.decoder = Decoders.HyperbolicGraphDecoder(
                latent_feature_dim=getattr(decoder_config, "latent_feature_dim", 10),
                hidden_feature_dim=getattr(decoder_config, "hidden_feature_dim", 32),
                output_feature_dim=getattr(decoder_config, "output_feature_dim", 3),
                num_decoder_layers=getattr(decoder_config, "num_layers", 3),
                layer_type=getattr(decoder_config, "layer_type", "HGAT"),
                dropout=getattr(decoder_config, "dropout", 0.0),
                edge_dim=getattr(decoder_config, "edge_dim", 1),
                normalization_factor=getattr(decoder_config, "normalization_factor", 1.0),
                aggregation_method=getattr(decoder_config, "aggregation_method", "sum"),
                message_transformation=getattr(decoder_config, "message_transformation", "linear"),
                aggregation_transformation=getattr(
                    decoder_config, "aggregation_transformation", "linear"
                ),
                use_normalization=getattr(decoder_config, "use_normalization", False),
                manifold_type=getattr(decoder_config, "manifold_type", "PoincareBall"),
                curvature=getattr(decoder_config, "curvature", 1.0),
                learnable_curvature=getattr(decoder_config, "learnable_curvature", False),
                use_centroid=getattr(decoder_config, "use_centroid", False),
                input_manifold=(
                    self.encoder.manifolds[-1] if hasattr(self.encoder, "manifolds") else None
                ),
            )
        elif decoder_type == "CentroidDistanceDecoder":
            self.decoder = Decoders.CentroidDistanceDecoder(
                model_dim=getattr(decoder_config, "hidden_feature_dim", 32),
                num_classes=getattr(decoder_config, "num_classes", 3),
                classifier_dropout=getattr(decoder_config, "dropout", 0.0),
            )
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

    def _initialize_edge_prediction(self, edge_predictor_config):
        """Initialize edge prediction components based on configuration."""
        if edge_predictor_config and hasattr(edge_predictor_config, "type"):
            if edge_predictor_config.type == "FermiDiracDecoder":
                self.edge_predictor = FermiDiracDecoder(self.encoder.manifold)
                loss_type = getattr(edge_predictor_config, "loss_type", "CrossEntropyLoss")
                loss_reduction = getattr(edge_predictor_config, "loss_reduction", "mean")
                if loss_type == "CrossEntropyLoss":
                    self.edge_loss_fn = nn.CrossEntropyLoss(reduction=loss_reduction)
                else:
                    raise ValueError(f"Unsupported edge loss type: {loss_type}")
            else:
                raise ValueError(f"Unsupported edge predictor type: {edge_predictor_config.type}")
        else:
            # Default edge predictor
            self.edge_predictor = FermiDiracDecoder(self.encoder.manifold)
            self.edge_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def _initialize_graph_prototypes(self):
        """Initialize graph prototypes for prototype loss if enabled."""
        # For prototype loss, we still need a reasonable number of prototypes
        # Default to 3 prototypes unless we have more information
        num_prototypes = 3
        self.graph_prototypes = nn.Parameter(torch.randn(num_prototypes, self.latent_dim * 2))
        std = 1.0 / ((self.latent_dim * 2) ** 0.5)
        nn.init.normal_(self.graph_prototypes, mean=0.0, std=std)

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

        # Determine if we need to compute mean_graph (for base_proto_loss)
        if self.use_base_proto_loss:
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
