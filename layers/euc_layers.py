"""Euclidean layers."""
import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act(hidden_dim, act_name='ReLU', num_layers=3, enc=True):
    """
    Helper function to get dimension and activation at every layer for Euclidean models.
    
    Args:
        hidden_dim: Hidden layer dimension
        act_name: Activation function name (e.g., 'ReLU', 'LeakyReLU')
        num_layers: Number of layers
        enc: Whether this is for encoder (True) or decoder (False)
        
    Returns:
        dims: List of layer dimensions
        acts: List of activation functions
    """
    act_class = getattr(nn, act_name)
    if isinstance(act_class(), nn.LeakyReLU):
        acts = [act_class(0.5)] * num_layers
    else:
        acts = [act_class()] * num_layers

    if enc:
        dims = [hidden_dim] * (num_layers + 1)  # len=num_layers+1
    else:
        dims = [hidden_dim] * (num_layers + 1)  # len=num_layers+1

    return dims, acts


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GraphConvolutionLayer(nn.Module):
    """
    Euclidean Graph Convolution Layer.
    Improved version with clearer parameter naming and documentation.
    """
    
    def __init__(self, input_feature_dim, output_feature_dim, dropout=0.0, 
                 activation=nn.ReLU(), edge_feature_dim=0, normalization_factor=1.0,
                 aggregation_method='sum', use_message_transform=True, 
                 use_output_transform=True):
        super(GraphConvolutionLayer, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.linear = nn.Linear(input_feature_dim, output_feature_dim, bias=True)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        # Attention mechanism initialization  
        dense_att_edge_dim = 1  # From adj.unsqueeze(-1) as the third argument
        if self.edge_feature_dim > 0:
            dense_att_edge_dim += 1 # From adj.unsqueeze(-1) as the fourth argument
        self.attention = DenseAtt(output_feature_dim, dropout, edge_dim=dense_att_edge_dim)
        
        self.use_message_transform = use_message_transform
        self.use_output_transform = use_output_transform
        self.activation = activation
        
        # Message transformation network
        if use_message_transform:
            self.message_network = nn.Sequential(
                nn.Linear(output_feature_dim + 1, output_feature_dim),
                activation,
                nn.Linear(output_feature_dim, output_feature_dim)
            )
            
        # Output transformation network
        if use_output_transform:
            self.output_network = nn.Sequential(
                nn.Linear(output_feature_dim, output_feature_dim),
                activation,
                nn.Linear(output_feature_dim, output_feature_dim)
            )
            
        self.layer_norm = nn.LayerNorm(output_feature_dim)

    def forward(self, input_data):
        node_features, adjacency_matrix = input_data
        node_features = self.linear(node_features)
        node_features = self._aggregate_messages(node_features, adjacency_matrix)
        node_features = self.layer_norm(node_features)
        node_features = self.activation(node_features)
        return node_features, adjacency_matrix

    def _aggregate_messages(self, node_features, adjacency_matrix):
        """Aggregate messages from neighboring nodes."""
        batch_size, num_nodes, _ = node_features.size()
        
        # Prepare node features for message passing
        # node_features_left: b x n x 1 x d     0,0,...0,1,1,...1...
        node_features_left = torch.unsqueeze(node_features, 2)
        node_features_left = node_features_left.expand(-1, -1, num_nodes, -1)
        
        # node_features_right: b x 1 x n x d     0,1,...n-1,0,1,...n-1...
        node_features_right = torch.unsqueeze(node_features, 1)
        node_features_right = node_features_right.expand(-1, num_nodes, -1, -1)

        # Apply message transformation if enabled
        if self.use_message_transform:
            # Ensure adjacency matrix has correct batch dimensions
            adj_for_message = adjacency_matrix
            if adjacency_matrix.dim() == 2: # Input adj is (n, n)
                if batch_size == 1:
                    adj_for_message = adjacency_matrix.unsqueeze(0) # -> (1, n, n)
                else: # batch_size > 1
                    adj_for_message = adjacency_matrix.unsqueeze(0).expand(batch_size, num_nodes, num_nodes) # -> (b, n, n)
            
            # Apply message transformation
            message_input = torch.cat([node_features_right, adj_for_message.unsqueeze(-1)], dim=-1)
            transformed_messages = self.message_network(message_input)
        else:
            transformed_messages = node_features_right

        # Prepare attention inputs
        # 1. Adjacency matrix for attention (third parameter)
        processed_adj_for_attention = adjacency_matrix
        if adjacency_matrix.dim() == 2: # Input adj is (n, n)
            if batch_size == 1:
                processed_adj_for_attention = adjacency_matrix.unsqueeze(0) # -> (1, n, n)
            else: # batch_size > 1
                processed_adj_for_attention = adjacency_matrix.unsqueeze(0).expand(batch_size, num_nodes, num_nodes) # -> (b, n, n)
        
        adj_for_attention = processed_adj_for_attention.unsqueeze(-1) # -> (b, n, n, 1)

        # 2. Edge attributes for attention (fourth parameter)
        edge_attr_for_attention = None
        if self.edge_feature_dim > 0: 
            edge_attr_for_attention = processed_adj_for_attention.unsqueeze(-1) # -> (b, n, n, 1)
        
        # Compute attention weights and aggregate messages
        attention_weights = self.attention(node_features_left, node_features_right, adj_for_attention, edge_attr_for_attention)
        aggregated_messages = transformed_messages * attention_weights
        aggregated_messages = torch.sum(aggregated_messages, dim=2)
        
        # Apply output transformation if enabled
        if self.use_output_transform:
            aggregated_messages = self.output_network(aggregated_messages)
            
        # Residual connection
        updated_features = node_features + aggregated_messages
        return updated_features


# Backward compatibility alias
class GCLayer(GraphConvolutionLayer):
    """
    Backward compatibility alias for GraphConvolutionLayer.
    Maps old parameter names to new ones.
    """
    
    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(), edge_dim=0, 
                 normalization_factor=1, aggregation_method='sum', msg_transform=True, 
                 sum_transform=True):
        super(GCLayer, self).__init__(
            input_feature_dim=in_dim,
            output_feature_dim=out_dim,
            dropout=dropout,
            activation=act,
            edge_feature_dim=edge_dim,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            use_message_transform=msg_transform,
            use_output_transform=sum_transform
        )


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for Euclidean spaces.
    Improved version with clearer parameter naming and documentation.
    """

    def __init__(self, input_feature_dim, output_feature_dim, dropout=0.0, 
                 activation=nn.LeakyReLU(0.5), edge_feature_dim=0, normalization_factor=1.0,
                 aggregation_method='sum', use_message_transform=True, 
                 use_output_transform=True, normalization_type='ln', num_attention_heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.linear_projection = nn.Linear(input_feature_dim, output_feature_dim, bias=False)
        self.attention_scoring = nn.Linear(2 * output_feature_dim // num_attention_heads + 1, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.num_attention_heads = num_attention_heads
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bias = nn.Parameter(torch.Tensor(output_feature_dim))
        
        # Skip connection projection if dimensions don't match
        if input_feature_dim != output_feature_dim:
            self.skip_projection = nn.Linear(input_feature_dim, output_feature_dim, bias=False)
        else:
            self.skip_projection = None
            
        self.init_parameters()

    def init_parameters(self):
        """Initialize layer parameters."""
        nn.init.xavier_uniform_(self.linear_projection.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input_data):
        node_features, adjacency_matrix = input_data

        batch_size, num_nodes, _ = node_features.size()
        node_features = self.dropout(node_features)
        
        # Project node features and reshape for multi-head attention
        projected_features = self.linear_projection(node_features).view(
            batch_size, num_nodes, self.num_attention_heads, -1
        )  # (b, n, num_heads, output_dim/num_heads)
        projected_features = self.dropout(projected_features)
        
        # Prepare features for attention computation
        features_left = torch.unsqueeze(projected_features, 2)
        features_left = features_left.expand(-1, -1, num_nodes, -1, -1)
        features_right = torch.unsqueeze(projected_features, 1)
        features_right = features_right.expand(-1, num_nodes, -1, -1, -1)  # (b, n, n, num_heads, dim/num_heads)
        
        # Compute attention scores
        attention_input = torch.cat([
            features_left, 
            features_right,
            adjacency_matrix[..., None, None].expand(-1, -1, -1, self.num_attention_heads, -1)
        ], dim=-1)
        attention_scores = self.attention_scoring(attention_input).squeeze()
        attention_scores = self.leaky_relu(attention_scores)  # (b, n, n, num_heads)
        
        # Apply edge mask to attention scores
        edge_mask = (adjacency_matrix > 1e-5).float()
        padding_mask = 1 - edge_mask
        masked_scores = -9e15 * padding_mask  # (b, n, n)

        attention_scores = attention_scores + masked_scores.unsqueeze(-1).expand(-1, -1, -1, self.num_attention_heads)
        attention_weights = torch.softmax(attention_scores, dim=2).transpose(2, 3)  # (b, n, num_heads, n)
        attention_weights = self.dropout(attention_weights).transpose(2, 3).unsqueeze(-1)
        
        # Apply attention to aggregate messages
        messages = features_right * attention_weights  # (b, n, n, num_heads, dim/num_heads)
        aggregated_messages = torch.sum(messages, dim=2)  # (b, n, num_heads, dim/num_heads)
        
        # Apply skip connection if needed
        if self.input_feature_dim != self.output_feature_dim:
            node_features = self.skip_projection(node_features)  # (b, n, output_dim)

        # Combine original features with aggregated messages
        updated_features = node_features + aggregated_messages.view(batch_size, num_nodes, -1) + self.bias
        updated_features = self.activation(updated_features)
        return updated_features, adjacency_matrix


# Backward compatibility alias
class GATLayer(GraphAttentionLayer):
    """
    Backward compatibility alias for GraphAttentionLayer.
    Maps old parameter names to new ones.
    """

    def __init__(self, in_dim, out_dim, dropout=0., act=nn.LeakyReLU(0.5), edge_dim=0, 
                 normalization_factor=1, aggregation_method='sum', msg_transform=True, 
                 sum_transform=True, use_norm='ln', num_of_heads=4):
        super(GATLayer, self).__init__(
            input_feature_dim=in_dim,
            output_feature_dim=out_dim,
            dropout=dropout,
            activation=act,
            edge_feature_dim=edge_dim,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            use_message_transform=msg_transform,
            use_output_transform=sum_transform,
            normalization_type=use_norm,
            num_attention_heads=num_of_heads
        )

'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs


# Backward compatibility function
def get_dim_act_legacy(config, num_layers, enc=True):
    """
    Backward compatibility wrapper for get_dim_act.
    Uses config object to extract parameters.
    """
    model_config = config.model
    return get_dim_act(
        hidden_dim=model_config.hidden_dim,
        act_name=model_config.act,
        num_layers=num_layers,
        enc=enc
    )
