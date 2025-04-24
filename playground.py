import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import yaml
import ml_collections
import networkx as nx
import torch
import numpy as np # Add numpy import

from utils.data_utils import load_from_file

# Load configuration (using the example from data_utils.py)
yaml_config_path = (
    pathlib.Path(__file__).parent.resolve()
    / "configs"
    / "enzymes_configs"
    / "enzymes_train_score.yaml"
)
with open(yaml_config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config = ml_collections.ConfigDict(config_dict)

# Call load_from_file
g_list, label_dict, tagset, all_nx_graphs = load_from_file(config, degree_as_tag=True)

# Check if any graphs were loaded using all_nx_graphs
if all_nx_graphs:
    # Get the first graph directly from all_nx_graphs
    first_nx_graph = all_nx_graphs[0]
    adj_matrix_sparse = nx.adjacency_matrix(first_nx_graph)
    adj_matrix_dense = adj_matrix_sparse.todense()


    node_features_list = []
    for node_id in sorted(first_nx_graph.nodes()): # Ensure consistent node order
            feature_data = first_nx_graph.nodes[node_id]['feature']
            node_features_list.append(torch.from_numpy(feature_data))
    feature_matrix = torch.stack(node_features_list).float() # Ensure float type

    print("First Graph (from all_nx_graphs):")
    print("\nAdjacency Matrix (dense):")
    print(adj_matrix_dense)
    print("\nFeature Matrix (from node attributes):")
    print(feature_matrix)
    print(f"\nFeature Matrix Shape: {feature_matrix.shape}")
    print(f"Adjacency Matrix Shape: {adj_matrix_dense.shape}")

# Check if any graphs were loaded
if g_list: # Check g_list instead of all_nx_graphs
    # Get the first graph object
    first_graph_obj = g_list[0]
    # Get the networkx graph from the Graph object
    first_nx_graph = first_graph_obj.g

    # Get the adjacency matrix (as a SciPy sparse matrix)
    adj_matrix_sparse = nx.adjacency_matrix(first_nx_graph)
    # Convert to dense numpy array for printing if needed
    adj_matrix_dense = adj_matrix_sparse.todense()

    # Get the feature matrix (already a torch tensor)
    feature_matrix = first_graph_obj.node_features

    print("First Graph:")
    print("\nAdjacency Matrix (dense):")
    print(adj_matrix_dense)
    print("\nFeature Matrix:")
    print(feature_matrix)
    print(f"\nFeature Matrix Shape: {feature_matrix.shape}")
    print(f"Adjacency Matrix Shape: {adj_matrix_dense.shape}")

else:
    print("No graphs were loaded.")
