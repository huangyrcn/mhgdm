import torch
from geoopt import ManifoldParameter
from torch import nn
from utils.Distributions import DiagonalGaussianDistribution
from layers.hyp_layers import get_dim_act_curv, HGCLayer, HGATLayer
from layers.euc_layers import get_dim_act, GCLayer, GATLayer


def compute_coordinate_differences(node_features, edge_index):
    """Calculate coordinate differences between connected nodes."""
    row, col = edge_index
    coordinate_differences = node_features[row] - node_features[col]
    radial_distances = torch.sum((coordinate_differences) ** 2, 1).unsqueeze(1)
    return torch.sqrt(radial_distances + 1e-8)


class GraphEncoder(nn.Module):
    """
    Abstract base class for graph encoders.
    Encodes node features into latent representations using graph neural networks.
    """

    def __init__(self, input_feature_dim, hidden_feature_dim, latent_feature_dim):
        super(GraphEncoder, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.latent_feature_dim = latent_feature_dim

        # Embed input node features to hidden dimension
        self.feature_embedding = nn.Linear(self.input_feature_dim, self.hidden_feature_dim, bias=False)
        
        # Project hidden features to mean and log-variance for variational distribution
        self.latent_projection = nn.Linear(self.hidden_feature_dim, 2 * self.latent_feature_dim)

    def forward(self, node_features, adjacency_matrix, node_mask):
        """
        Forward pass through encoder.
        
        Args:
            node_features: Input node feature matrix [batch_size, num_nodes, input_feature_dim]
            adjacency_matrix: Graph adjacency matrix [batch_size, num_nodes, num_nodes]  
            node_mask: Mask for valid nodes [batch_size, num_nodes, 1]
            
        Returns:
            posterior: Diagonal Gaussian distribution in latent space
        """
        # Embed node features to hidden dimension
        embedded_features = self.feature_embedding(node_features)
        
        # Encode through graph convolution layers
        encoded_features = self.encode(embedded_features, adjacency_matrix)
        
        # Project to latent space (mean and log-variance)
        mean_logvar = self.latent_projection(encoded_features)
        posterior = DiagonalGaussianDistribution(mean_logvar, self.manifold, node_mask)
        
        return posterior

class EuclideanGraphEncoder(GraphEncoder):
    """
    Euclidean Graph Convolutional Network Encoder.
    Uses standard GCN or GAT layers operating in Euclidean space.
    """

    def __init__(self, input_feature_dim, hidden_feature_dim, latent_feature_dim, 
                 num_encoder_layers, layer_type='GCN', dropout=0.0, edge_dim=1,
                 normalization_factor=1.0, aggregation_method='sum', 
                 message_transformation='linear'):
        super(EuclideanGraphEncoder, self).__init__(input_feature_dim, hidden_feature_dim, latent_feature_dim)
        self.num_encoder_layers = num_encoder_layers
        
        # Get layer dimensions and activation functions
        self.layer_dimensions, self.activations = get_dim_act(
            hidden_dim=self.hidden_feature_dim, 
            act_name='ReLU',
            num_layers=self.num_encoder_layers
        )
        self.manifold = None  # Euclidean space doesn't use manifolds
        
        # Build graph convolution layers
        graph_layers = []
        layer_class = self._get_layer_class(layer_type)
        
        for i in range(self.num_encoder_layers):
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
                    sum_transform=message_transformation  # Use same transform for consistency
                )
            )
        
        self.graph_layers = nn.Sequential(*graph_layers)

    def _get_layer_class(self, layer_type):
        """Get the appropriate layer class based on layer type."""
        if layer_type == 'GCN':
            return GCLayer
        elif layer_type == 'GAT':
            return GATLayer
        else:
            raise ValueError(f"Unknown layer type: {layer_type}. Supported types: ['GCN', 'GAT']")

    def encode(self, node_features, adjacency_matrix):
        """Encode node features through Euclidean graph layers."""
        encoded_features, _ = self.graph_layers((node_features, adjacency_matrix))
        return encoded_features


class HyperbolicGraphEncoder(GraphEncoder):
    """
    Hyperbolic Graph Convolutional Network Encoder.
    Uses HGCN or HGAT layers operating in hyperbolic manifolds (Poincaré Ball or Lorentz).
    """

    def __init__(self, input_feature_dim, hidden_feature_dim, latent_feature_dim,
                 num_encoder_layers, layer_type='HGCN', dropout=0.0, edge_dim=1,
                 normalization_factor=1.0, aggregation_method='sum', 
                 message_transformation='linear', aggregation_transformation='linear',
                 use_normalization=False, manifold_type='PoincareBall', 
                 curvature=1.0, learnable_curvature=False):
        super(HyperbolicGraphEncoder, self).__init__(input_feature_dim, hidden_feature_dim, latent_feature_dim)
        self.num_encoder_layers = num_encoder_layers
        
        # Get layer dimensions, activations, and manifolds for hyperbolic layers
        self.layer_dimensions, self.activations, self.manifolds = get_dim_act_curv(
            hidden_dim=self.hidden_feature_dim,
            dim=self.latent_feature_dim, 
            manifold_name=manifold_type,
            c=curvature,
            learnable_c=learnable_curvature,
            act_name='ReLU',
            num_layers=self.num_encoder_layers,
            enc=True  # This is an encoder
        )
        self.manifold = self.manifolds[-1]  # Final manifold for output
        
        # Build hyperbolic graph convolution layers
        hyperbolic_layers = []
        layer_class = self._get_layer_class(layer_type)
        
        for i in range(self.num_encoder_layers):
            input_manifold, output_manifold = self.manifolds[i], self.manifolds[i + 1]
            input_dim, output_dim = self.layer_dimensions[i], self.layer_dimensions[i + 1]
            activation = self.activations[i]
            
            hyperbolic_layers.append(
                layer_class(
                    input_dim, output_dim, 
                    input_manifold, output_manifold, 
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
        
        self.graph_layers = nn.Sequential(*hyperbolic_layers)
        
        # Initialize embedding weights on the hyperbolic manifold
        self._initialize_hyperbolic_embedding()

    def _get_layer_class(self, layer_type):
        """Get the appropriate hyperbolic layer class based on layer type."""
        if layer_type == 'HGCN':
            return HGCLayer
        elif layer_type == 'HGAT':
            return HGATLayer
        else:
            raise ValueError(f"Unknown hyperbolic layer type: {layer_type}. Supported types: ['HGCN', 'HGAT']")

    def _initialize_hyperbolic_embedding(self):
        """Initialize embedding weights on the hyperbolic manifold."""
        std = 1 / torch.sqrt(torch.abs(self.manifold.k))
        manifold_weights = self.manifolds[0].random_normal(
            (self.input_feature_dim, self.hidden_feature_dim), 
            std=std
        ).T
        self.feature_embedding.weight = ManifoldParameter(manifold_weights, self.manifolds[0])

    def encode(self, node_features, adjacency_matrix):
        """Encode node features through hyperbolic graph layers."""
        # Forward pass through hyperbolic layers
        encoded_features, _ = self.graph_layers((node_features, adjacency_matrix))
        
        # Map from hyperbolic space to tangent space at origin for variational distribution
        tangent_features = self.manifolds[-1].logmap0(encoded_features)
        
        return tangent_features

    def project_to_tangent_space(self, hyperbolic_tensor, manifold):
        """Project hyperbolic tensor to tangent space at origin."""
        if manifold.name == 'Lorentz':
            # For Lorentz model, project by zeroing the time component
            time_component = hyperbolic_tensor.narrow(-1, 0, 1)
            zeros = torch.zeros_like(hyperbolic_tensor)
            zeros[:, 0:1] = time_component
            return hyperbolic_tensor - zeros
        else:
            return hyperbolic_tensor


# Backward compatibility aliases
class GCN(EuclideanGraphEncoder):
    """
    Backward compatibility alias for EuclideanGraphEncoder.
    Graph Convolution Networks operating in Euclidean space.
    """
    
    def __init__(self, input_feat_dim, hidden_dim, dim, enc_layers, layer_type='GCN', 
                 dropout=0.0, edge_dim=1, normalization_factor=1.0, 
                 aggregation_method='sum', msg_transform='linear'):
        super(GCN, self).__init__(
            input_feature_dim=input_feat_dim,
            hidden_feature_dim=hidden_dim,
            latent_feature_dim=dim,
            num_encoder_layers=enc_layers,
            layer_type=layer_type,
            dropout=dropout,
            edge_dim=edge_dim,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            message_transformation=msg_transform
        )


class HGCN(HyperbolicGraphEncoder):
    """
    Backward compatibility alias for HyperbolicGraphEncoder.
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, input_feat_dim, hidden_dim, dim, enc_layers, layer_type='HGCN',
                 dropout=0.0, edge_dim=1, normalization_factor=1.0, 
                 aggregation_method='sum', msg_transform='linear', sum_transform='linear',
                 use_norm=False, manifold='PoincareBall', c=1.0, learnable_c=False):
        super(HGCN, self).__init__(
            input_feature_dim=input_feat_dim,
            hidden_feature_dim=hidden_dim,
            latent_feature_dim=dim,
            num_encoder_layers=enc_layers,
            layer_type=layer_type,
            dropout=dropout,
            edge_dim=edge_dim,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            message_transformation=msg_transform,
            aggregation_transformation=sum_transform,
            use_normalization=use_norm,
            manifold_type=manifold,
            curvature=c,
            learnable_curvature=learnable_c
        )
