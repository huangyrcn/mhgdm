import torch
import torch.nn as nn
from torch.nn import init

from layers.hyp_layers import HGCLayer, get_dim_act_curv, HGATLayer
from layers.euc_layers import GCLayer, get_dim_act, GATLayer
from layers.CentroidDistance import CentroidDistance
from utils import manifolds_utils


class GraphDecoder(nn.Module):
    """ 
    Abstract base class for graph decoders.
    Decodes latent representations back to node features.
    """

    def __init__(self, latent_feature_dim, hidden_feature_dim, output_feature_dim):
        super(GraphDecoder, self).__init__()
        self.latent_feature_dim = latent_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.output_feature_dim = output_feature_dim
        
        # Project decoded features to output node features
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_feature_dim, self.output_feature_dim),
        )

    def forward(self, latent_features, adjacency_matrix, node_mask):
        """
        Forward pass through decoder.
        
        Args:
            latent_features: Latent node representations [batch_size, num_nodes, latent_dim]
            adjacency_matrix: Graph adjacency matrix [batch_size, num_nodes, num_nodes]
            node_mask: Mask for valid nodes [batch_size, num_nodes, 1]
            
        Returns:
            node_predictions: Predicted node features [batch_size, num_nodes, feature_dim]
        """
        # Decode latent features back to hidden dimension
        decoded_features = self.decode(latent_features, adjacency_matrix)
        
        # Project to output node features and apply mask
        node_predictions = self.output_projection(decoded_features) * node_mask
        
        return node_predictions


class EuclideanGraphDecoder(GraphDecoder):
    """
    Euclidean Graph Convolutional Network Decoder.
    Uses standard GCN or GAT layers in Euclidean space.
    """

    def __init__(self, latent_feature_dim, hidden_feature_dim, output_feature_dim,
                 num_decoder_layers, layer_type='GCN', dropout=0.0, edge_dim=1,
                 normalization_factor=1.0, aggregation_method='sum', 
                 message_transformation='linear'):
        super(EuclideanGraphDecoder, self).__init__(latent_feature_dim, hidden_feature_dim, output_feature_dim)
        self.num_decoder_layers = num_decoder_layers
        
        # Get layer dimensions and activation functions
        self.layer_dimensions, self.activations = get_dim_act(
            hidden_dim=self.hidden_feature_dim, 
            act_name='ReLU',
            num_layers=self.num_decoder_layers,
            enc=False
        )
        
        # Override first layer dimension for proper decoder behavior
        # First layer: latent_feature_dim -> hidden_feature_dim
        # The last layer should output hidden_feature_dim, then output_projection maps to output_feature_dim
        self.layer_dimensions[0] = self.latent_feature_dim
        
        self.manifolds = None  # Euclidean space doesn't use manifolds
        
        # Build graph convolution layers
        graph_layers = []
        layer_class = self._get_layer_class(layer_type)
        
        for i in range(self.num_decoder_layers):
            input_dim, output_dim = self.layer_dimensions[i], self.layer_dimensions[i + 1]
            activation = self.activations[i]
            
            graph_layers.append(
                layer_class(
                    input_dim, output_dim,
                    dropout=dropout, 
                    act=activation, 
                    edge_dim=edge_dim,
                    normalization_factor=normalization_factor, 
                    aggregation_method=aggregation_method,
                    msg_transform=message_transformation, 
                    sum_transform=message_transformation
                )
            )
        
        self.graph_layers = nn.Sequential(*graph_layers)
        self.message_passing = True

    def _get_layer_class(self, layer_type):
        """Get the appropriate layer class based on layer type."""
        if layer_type == 'GCN':
            return GCLayer
        elif layer_type == 'GAT':
            return GATLayer
        else:
            raise ValueError(f"Unknown layer type: {layer_type}. Supported types: ['GCN', 'GAT']")

    def decode(self, latent_features, adjacency_matrix):
        """Decode latent features through Euclidean graph layers."""
        decoded_features = self.graph_layers((latent_features, adjacency_matrix))[0]
        return decoded_features

class HyperbolicGraphDecoder(GraphDecoder):
    """
    Hyperbolic Graph Convolutional Network Decoder.
    Uses HGCN or HGAT layers operating in hyperbolic manifolds.
    """

    def __init__(self, latent_feature_dim, hidden_feature_dim, output_feature_dim,
                 num_decoder_layers, layer_type='HGCN', dropout=0.0, edge_dim=1,
                 normalization_factor=1.0, aggregation_method='sum', 
                 message_transformation='linear', aggregation_transformation='linear',
                 use_normalization=False, manifold_type='PoincareBall', 
                 curvature=1.0, learnable_curvature=False, use_centroid=False,
                 input_manifold=None):
        super(HyperbolicGraphDecoder, self).__init__(latent_feature_dim, hidden_feature_dim, output_feature_dim)
        self.num_decoder_layers = num_decoder_layers
        self.use_centroid = use_centroid
        
        # Get layer dimensions, activations, and manifolds for hyperbolic layers
        self.layer_dimensions, self.activations, self.manifolds = get_dim_act_curv(
            hidden_dim=self.hidden_feature_dim,
            dim=self.latent_feature_dim, 
            manifold_name=manifold_type,
            c=curvature,
            learnable_c=learnable_curvature,
            act_name='ReLU',
            num_layers=self.num_decoder_layers,
            enc=False  # This is a decoder
        )
        
        # Override first layer dimension for proper decoder behavior
        # First layer: latent_feature_dim -> hidden_feature_dim
        # The last layer should output hidden_feature_dim, then output_projection maps to output_feature_dim
        self.layer_dimensions[0] = self.latent_feature_dim
        
        # Use provided input manifold or the computed one
        if input_manifold is not None:
            self.manifolds[0] = input_manifold
        self.manifold = self.manifolds[-1]
        
        # Build hyperbolic graph convolution layers
        hyperbolic_layers = []
        layer_class = self._get_layer_class(layer_type)
        
        for i in range(self.num_decoder_layers):
            input_manifold_layer, output_manifold_layer = self.manifolds[i], self.manifolds[i + 1]
            input_dim, output_dim = self.layer_dimensions[i], self.layer_dimensions[i + 1]
            activation = self.activations[i]
            
            hyperbolic_layers.append(
                layer_class(
                    input_dim, output_dim, input_manifold_layer, output_manifold_layer, 
                    dropout=dropout, 
                    act=activation, 
                    edge_dim=edge_dim,
                    normalization_factor=normalization_factor, 
                    aggregation_method=aggregation_method,
                    msg_transform=message_transformation, 
                    sum_transform=aggregation_transformation, 
                    use_norm=use_normalization
                )
            )
        
        # Optional centroid distance layer
        if self.use_centroid:
            self.centroid_layer = CentroidDistance(
                self.layer_dimensions[-1], self.layer_dimensions[-1], 
                self.manifold, dropout
            )
        
        self.graph_layers = nn.Sequential(*hyperbolic_layers)
        self.message_passing = True

    def _get_layer_class(self, layer_type):
        """Get the appropriate hyperbolic layer class based on layer type."""
        if layer_type == 'HGCN':
            return HGCLayer
        elif layer_type == 'HGAT':
            return HGATLayer
        else:
            raise ValueError(f"Unknown hyperbolic layer type: {layer_type}. Supported types: ['HGCN', 'HGAT']")

    def decode(self, latent_features, adjacency_matrix):
        """Decode latent features through hyperbolic graph layers."""
        # Forward pass through hyperbolic layers
        decoded_features, _ = self.graph_layers((latent_features, adjacency_matrix))
        
        # Apply centroid distance or map to tangent space
        if self.use_centroid:
            output = self.centroid_layer(decoded_features)
        else:
            # Map from hyperbolic space to tangent space at origin
            output = self.manifolds[-1].logmap0(decoded_features)
        
        return output

class CentroidDistanceDecoder(GraphDecoder):
    """
    Decoder using centroid distance computation.
    Specialized for hyperbolic manifolds.
    """

    def __init__(self, latent_feature_dim, hidden_feature_dim, output_feature_dim,
                 dropout=0.0, manifold=None):
        super(CentroidDistanceDecoder, self).__init__(latent_feature_dim, hidden_feature_dim, output_feature_dim)
        self.manifold = manifold
        self.centroid_layer = CentroidDistance(
            self.hidden_feature_dim, self.latent_feature_dim, 
            self.manifold, dropout
        )
        self.message_passing = True

    def decode(self, latent_features, adjacency_matrix):
        """Decode using centroid distance computation."""
        decoded_features = self.centroid_layer(latent_features)
        return decoded_features


class Classifier(nn.Module):
    """
    Graph-level classifier for few-shot learning tasks.
    Operates on concatenated node embeddings (mean + pooled representation).
    """
    
    def __init__(self, model_dim, num_classes, classifier_dropout=0.0, classifier_bias=True, manifold=None):
        super().__init__()
        self.manifold = manifold
        self.model_dim = model_dim
        self.num_classes = num_classes
        
        # Input dimension is model_dim * 2 (concatenated mean + pooled embeddings)
        input_dim = model_dim * 2
        
        # Build classifier layers
        layers = []
        if classifier_dropout > 0.0:
            layers.append(nn.Dropout(p=classifier_dropout))
        layers.append(nn.Linear(input_dim, num_classes, bias=classifier_bias))
        
        self.cls = nn.Sequential(*layers)

    def decode(self, h, adj=None):
        """
        Classify graph representation.
        
        Args:
            h: Concatenated graph embeddings [batch_size, model_dim * 2]
            adj: Adjacency matrix (not used, kept for interface consistency)
            
        Returns:
            class_logits: Classification logits [batch_size, num_classes]
        """
        # h is already the concatenated representation (mean + pooled)
        processed_h = h
        
        # If operating in hyperbolic space, project to tangent space
        if self.manifold is not None:
            processed_h = manifolds_utils.proj_tan0(h, self.manifold)
        
        return self.cls(processed_h)

    def forward(self, h, adj=None):
        """Forward pass through classifier."""
        return self.decode(h, adj)


class FermiDiracDecoder(nn.Module):
    """
    Fermi-Dirac Decoder for edge prediction.
    
    Computes edge probabilities based on distances between node representations
    using a Fermi-Dirac distribution. Supports multiple edge types.
    """

    def __init__(self, manifold):
        super(FermiDiracDecoder, self).__init__()
        self.manifold = manifold
        # Parameters for Fermi-Dirac distribution (3 edge types)
        self.r = nn.Parameter(torch.ones((3,), dtype=torch.float))  # Distance thresholds
        self.t = nn.Parameter(torch.ones((3,), dtype=torch.float))  # Temperature parameters

    def forward(self, x):
        """
        Forward pass for edge prediction.
        
        Args:
            x: Node representations [batch_size, num_nodes, feature_dim]
            
        Returns:
            edge_type: Edge type probabilities [batch_size, num_nodes, num_nodes, 4]
                      Last dimension: [no_edge, edge_type_1, edge_type_2, edge_type_3]
        """
        b, n, _ = x.size()
        
        # Compute pairwise representations
        x_left = x[:, :, None, :]      # [B, N, 1, D]
        x_right = x[:, None, :, :]     # [B, 1, N, D]
        
        # Compute pairwise distances
        if self.manifold is not None:
            dist = self.manifold.dist(x_left, x_right, keepdim=True)  # [B, N, N, 1]
        else:
            dist = torch.pairwise_distance(x_left, x_right, keepdim=True)  # [B, N, N, 1]
        
        # Apply Fermi-Dirac distribution for each edge type
        edge_type = 1.0 / (
            torch.exp((dist - self.r[None, None, None, :]) * self.t[None, None, None, :]) + 1.0
        )  # [B, N, N, 3]
        
        # Compute no-edge probability (1 - max edge type probability)
        no_edge = 1.0 - edge_type.max(dim=-1, keepdim=True)[0]  # [B, N, N, 1]
        
        # Concatenate: [no_edge, edge_type_1, edge_type_2, edge_type_3]
        edge_type = torch.cat([no_edge, edge_type], dim=-1)  # [B, N, N, 4]
        
        return edge_type


# Backward compatibility aliases
class EuclideanGCNDecoder(EuclideanGraphDecoder):
    """
    Backward compatibility alias for EuclideanGraphDecoder.
    """
    
    def __init__(self, config):
        super(EuclideanGCNDecoder, self).__init__(
            latent_feature_dim=config.model.dim,
            hidden_feature_dim=config.model.hidden_dim,
            output_feature_dim=config.data.max_feat_num,
            num_decoder_layers=config.model.dec_layers,
            layer_type=config.model.layer_type,
            dropout=config.model.dropout,
            edge_dim=config.model.edge_dim,
            normalization_factor=config.model.normalization_factor,
            aggregation_method=config.model.aggregation_method,
            message_transformation=config.model.msg_transform
        )


class HyperbolicGCNDecoder(HyperbolicGraphDecoder):
    """
    Backward compatibility alias for HyperbolicGraphDecoder.
    """

    def __init__(self, config, manifold=None):
        super(HyperbolicGCNDecoder, self).__init__(
            latent_feature_dim=config.model.dim,
            hidden_feature_dim=config.model.hidden_dim,
            output_feature_dim=config.data.max_feat_num,
            num_decoder_layers=config.model.dec_layers,
            layer_type=config.model.layer_type,
            dropout=config.model.dropout,
            edge_dim=config.model.edge_dim,
            normalization_factor=config.model.normalization_factor,
            aggregation_method=config.model.aggregation_method,
            message_transformation=config.model.msg_transform,
            aggregation_transformation=config.model.sum_transform,
            use_normalization=config.model.use_norm,
            manifold_type=config.model.manifold,
            curvature=config.model.c,
            learnable_curvature=config.model.learnable_c,
            use_centroid=config.model.use_centroid,
            input_manifold=manifold
        )


# Legacy alias for backward compatibility
Decoder = GraphDecoder