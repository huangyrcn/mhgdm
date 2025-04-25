import os
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
import pickle
import networkx as nx
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils.graph_utils import init_features, graphs_to_tensor
import json
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import node_flags


def load_data(config):

    dataset = MyDataset(config)
    # 返回 MyDataset 的训练和测试数据加载器
    return dataset.get_loaders()


class Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


import os
import torch
import networkx as nx
import numpy as np
import pickle


def load_from_file(config, degree_as_tag):
    file_path = os.path.join("./datasets", config.data.name, f"{config.data.name}.txt")
    save_file_path = os.path.join(
        "./datasets", config.data.name, f"{config.data.name}_processed.pkl"
    )

    if os.path.exists(save_file_path):
        print(f"Loading processed data from: {save_file_path}")
        with open(save_file_path, "rb") as f:
            return pickle.load(f)

    print(f"Loading data from: {file_path}")
    g_list = []
    label_dict = {}
    feat_dict = {}
    all_nx_graphs = []

    for_graph_node_counts = []
    for_graph_feat_dims = []

    with open(file_path, "r") as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]

            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped

            g = nx.Graph()
            node_tags = []
            node_features = []

            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])

                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if attr is not None:
                    node_features.append(attr)
                    g.nodes[j]["feature"] = attr
                else:
                    g.nodes[j]["feature"] = feat_dict[row[0]]

                for k in range(2, tmp):
                    g.add_edge(j, row[k])

            if node_features:
                node_features = np.stack(node_features)
                for_graph_feat_dims.append(node_features.shape[1])
            else:
                for_graph_feat_dims.append(1)  # e.g., scalar tag

            assert len(g) == n
            for_graph_node_counts.append(n)

            g_list.append(Graph(g, l, node_tags))
            all_nx_graphs.append(g)

    for graph_obj in g_list:
        graph_obj.neighbors = [[] for _ in range(len(graph_obj.g))]
        for i, j in graph_obj.g.edges():
            graph_obj.neighbors[i].append(j)
            graph_obj.neighbors[j].append(i)

        degree_list = [len(neighbors) for neighbors in graph_obj.neighbors]
        graph_obj.max_neighbor = max(degree_list) if degree_list else 0

        edges = [list(pair) for pair in graph_obj.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        graph_obj.edge_mat = (
            torch.LongTensor(edges).transpose(0, 1) if edges else torch.LongTensor(2, 0)
        )

    if degree_as_tag:
        for graph_obj in g_list:
            graph_obj.node_tags = [d for n, d in graph_obj.g.degree()]
            if np.sum(np.array(graph_obj.node_tags) == 0) > 0:
                print(f"Graph with isolated node found. Degrees: {graph_obj.node_tags}")

    tagset = sorted(set(tag for g in g_list for tag in g.node_tags))
    tag2index = {tag: i for i, tag in enumerate(tagset)}

    for i, graph_obj in enumerate(g_list):
        num_nodes = len(graph_obj.node_tags)
        node_indices = [tag2index[tag] for tag in graph_obj.node_tags]
        graph_obj.node_features = torch.zeros(num_nodes, len(tagset))
        graph_obj.node_features[torch.arange(num_nodes), node_indices] = 1

        nx_graph = all_nx_graphs[i]
        for node_idx in range(num_nodes):
            if node_idx in nx_graph.nodes:
                nx_graph.nodes[node_idx]["feature"] = graph_obj.node_features[node_idx].numpy()
            else:
                print(
                    f"Warning: Node {node_idx} not found in nx_graph {i} while assigning features."
                )

    # 计算 max_node_num 和 max_feat_num
    max_node_num = max(for_graph_node_counts) if for_graph_node_counts else 0  # Handle empty list
    max_feat_num = max(for_graph_feat_dims) if for_graph_feat_dims else 0  # Handle empty list

    # If degrees are used as tags, the feature dimension is the number of unique tags
    if degree_as_tag:
        max_feat_num = len(tagset)

    # Save processed data including the potentially updated max_feat_num
    with open(save_file_path, "wb") as f:
        # Save all necessary components including the final max values
        pickle.dump((g_list, label_dict, tagset, all_nx_graphs, max_node_num, max_feat_num), f)

    return g_list, label_dict, tagset, all_nx_graphs, max_node_num, max_feat_num


class MyDataset:
    def __init__(self, config):
        self.dataset_name = config.data.name
        self.config = config
        self.train_graphs = []
        self.test_graphs = []
        self.train_nx_graphs = []
        self.test_nx_graphs = []

        (
            all_graphs,
            label_dict,
            tagset,
            all_nx_graphs,
            self.config.data.max_node_num,
            self.config.data.max_feat_num,
        ) = load_from_file(config=self.config, degree_as_tag=True)

        # Calculate and store max_node_num and max_feat_num
        self.max_node_num = max(len(g.nodes) for g in all_nx_graphs) if all_nx_graphs else 0
        self.tagset = tagset  # Store tagset
        self.max_feat_num = len(
            self.tagset
        )  # Feature dimension is based on the number of unique tags (degrees)

        with open("./datasets/{}/train_test_classes.json".format(config.data.name), "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]
            self.test_classes = all_class_splits["test"]

        train_classes_mapping = {}
        for cl in self.train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        self.train_classes_num = len(train_classes_mapping)

        test_classes_mapping = {}
        for cl in self.test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        for i in range(len(all_graphs)):
            original_label = all_graphs[i].label
            if original_label in self.train_classes:
                all_graphs[i].label = train_classes_mapping[int(original_label)]
                self.train_graphs.append(all_graphs[i])
                self.train_nx_graphs.append(all_nx_graphs[i])
                all_nx_graphs[i].graph["label"] = all_graphs[i].label

            if original_label in self.test_classes:
                all_graphs[i].label = test_classes_mapping[int(original_label)]
                self.test_graphs.append(all_graphs[i])
                self.test_nx_graphs.append(all_nx_graphs[i])
                all_nx_graphs[i].graph["label"] = all_graphs[i].label

        np.random.shuffle(self.train_graphs)
        self.train_tasks = defaultdict(list)
        for graph in self.train_graphs:
            self.train_tasks[graph.label].append(graph)

        np.random.shuffle(self.test_graphs)
        self.test_tasks = defaultdict(list)
        for graph in self.test_graphs:
            self.test_tasks[graph.label].append(graph)

        self.total_test_g_list = []
        for index in range(self.test_classes_num):
            query_pool_for_class = list(self.test_tasks[index])[self.config.fsl_task.K_shot :]
            self.total_test_g_list.extend(query_pool_for_class)

        from numpy.random import RandomState

        rd = RandomState(0)
        rd.shuffle(self.total_test_g_list)

    def get_loaders(self):
        def make_loader(nx_graphs):
            adjs_tensor = graphs_to_tensor(nx_graphs, self.config.data.max_node_num)
            x_tensor = init_features(
                self.config.data.init, adjs_tensor
            )

            labels_tensor = torch.LongTensor([nx_g.graph["label"] for nx_g in nx_graphs])
            dataset = TensorDataset(x_tensor, adjs_tensor, labels_tensor)
            loader = DataLoader(
                dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                num_workers=8,  # Or a higher value based on your CPU cores
                pin_memory=True,
            )
            return loader

        train_loader = make_loader(self.train_nx_graphs)
        test_loader = make_loader(self.test_nx_graphs)
        return train_loader, test_loader

    def sample_P_tasks(self, task_source, P_num_task, sample_rate, N_way, K_shot, query_size):
        tasks = []
        support_classes = []
        num_available_classes = sample_rate.shape[0]

        for _ in range(P_num_task):
            chosen_class_indices = np.random.choice(
                list(range(num_available_classes)), N_way, p=sample_rate, replace=False
            )
            support_classes.append(chosen_class_indices)
            tasks.append(
                self.sample_one_task(
                    task_source, chosen_class_indices, K_shot=K_shot, query_size=query_size
                )
            )

        return tasks, support_classes

    def sample_one_task(self, task_source, class_index, K_shot, query_size, test_start_idx=None):
        support_set = []
        query_set = []

        for index in class_index:
            g_list = list(task_source[index])
            if task_source == self.train_tasks or test_start_idx is None:
                np.random.shuffle(g_list)
            support_set.append(g_list[:K_shot])
            if task_source == self.train_tasks or test_start_idx is None:
                query_set.append(g_list[K_shot : K_shot + query_size])

        append_count = 0
        if task_source == self.test_tasks and test_start_idx is not None:
            query_set = []
            num_classes_in_task = len(class_index)
            for i in range(num_classes_in_task):
                start = min(test_start_idx + i * query_size, len(self.total_test_g_list))
                end = min(test_start_idx + (i + 1) * query_size, len(self.total_test_g_list))
                current_query_graphs = self.total_test_g_list[start:end]
                while len(current_query_graphs) < query_size:
                    if current_query_graphs:
                        current_query_graphs.append(current_query_graphs[-1])
                    elif support_set and support_set[0]:
                        current_query_graphs.append(support_set[0][-1])
                    else:
                        print(
                            f"Warning: Cannot fill query set for class {class_index[i]} - insufficient data."
                        )
                        break
                    append_count += 1
                query_set.append(current_query_graphs)

        return {"support_set": support_set, "query_set": query_set, "append_count": append_count}
    

