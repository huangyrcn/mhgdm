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
from numpy.random import RandomState


def load_data(config, get_graph_list=False):

    dataset = MyDataset(config)
    if get_graph_list:
        return dataset.train_graphs, dataset.test_graphs
    else:

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

        (
            all_graphs,
            label_dict,
            tagset,
            all_nx_graphs,
            max_node_num,
            max_feat_num,
        ) = load_from_file(config=self.config, degree_as_tag=True)

        self.max_node_num = self.config.data.max_node_num
        self.max_feat_num = self.config.data.max_feat_num
        self.batch_size = self.config.data.batch_size
        self.tagset = tagset

        with open(f"./datasets/{config.data.name}/train_test_classes.json", "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]
            self.test_classes = all_class_splits["test"]

        train_classes_mapping = {cl: idx for idx, cl in enumerate(self.train_classes)}
        test_classes_mapping = {cl: idx for idx, cl in enumerate(self.test_classes)}

        self.train_classes_num = len(train_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        self.train_graphs = []
        self.test_graphs = []
        train_labels = []
        test_labels = []

        for i in range(len(all_graphs)):
            original_label = all_graphs[i].label
            if original_label in self.train_classes:
                all_graphs[i].label = train_classes_mapping[int(original_label)]
                self.train_graphs.append(all_nx_graphs[i])
                train_labels.append(all_graphs[i].label)
            elif original_label in self.test_classes:
                all_graphs[i].label = test_classes_mapping[int(original_label)]
                self.test_graphs.append(all_nx_graphs[i])
                test_labels.append(all_graphs[i].label)
        # 转 tensor
        self.train_adjs, self.train_x = graphs_to_tensor(
            self.train_graphs, self.max_node_num, self.max_feat_num
        )
        self.test_adjs, self.test_x = graphs_to_tensor(
            self.test_graphs, self.max_node_num, self.max_feat_num
        )
        self.train_labels = torch.LongTensor(train_labels)
        self.test_labels = torch.LongTensor(test_labels)

        # 只有有 fsl_task 配置时，才执行 few-shot 相关代码
        if not hasattr(self.config, "fsl_task"):
            print("未检测到 fsl_task 配置，跳过 few-shot 相关初始化。")
            return

        # 组织成 task 列表 (Graph对象用于sample，后续也可以升级为tensor版)
        self.train_tasks = defaultdict(list)
        for graph, label in zip(all_graphs, train_labels):
            self.train_tasks[label].append(graph)

        self.test_tasks = defaultdict(list)
        for graph, label in zip(all_graphs, test_labels):
            self.test_tasks[label].append(graph)

        # 组织成 tensor-based task 列表
        self.train_tasks_tensor = defaultdict(list)
        self.train_indices = defaultdict(list)
        for idx, label in enumerate(self.train_labels):
            self.train_indices[label.item()].append(idx)

        self.test_tasks_tensor = defaultdict(list)
        self.test_indices = defaultdict(list)
        for idx, label in enumerate(self.test_labels):
            self.test_indices[label.item()].append(idx)

        # 测试用的全部 test graph (用于few-shot query pool)
        self.total_test_g_list = []
        for index in range(self.test_classes_num):
            query_pool_for_class = list(self.test_tasks[index])[self.config.fsl_task.K_shot :]
            self.total_test_g_list.extend(query_pool_for_class)

        # 测试用的全部 test indices (用于tensor版few-shot query pool)
        self.total_test_indices = []
        for index in range(self.test_classes_num):
            query_indices = self.test_indices[index][self.config.fsl_task.K_shot :]
            self.total_test_indices.extend(query_indices)

        rd = RandomState(0)
        rd.shuffle(self.total_test_g_list)
        rd.seed(0)  # Reset seed to ensure same shuffling pattern
        rd.shuffle(self.total_test_indices)

    def get_loaders(self):
        train_dataset = TensorDataset(self.train_x, self.train_adjs, self.train_labels)
        test_dataset = TensorDataset(self.test_x, self.test_adjs, self.test_labels)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=8,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=8,
        )
        return train_loader, test_loader

    def sample_one_task(self, is_train, N_way, K_shot, R_query, test_start_idx=None):
        """
        Sample one task for few-shot learning, returning tensor data directly.

        Args:
            is_train: 是否为训练集任务 (True/False)
            N_way: 任务类别数
            K_shot: 每类支持集样本数
            R_query: 每类查询集样本数
            test_start_idx: 测试查询集采样起始索引（仅测试时用）

        Returns:
            dict: A dictionary containing:
                - "support_set": Dict with tensors "x", "adj", "label" for support set
                - "query_set": Dict with tensors "x", "adj", "label" for query set
                - "append_count": Number of times samples were appended
        """
        # 根据 is_train 选择数据源
        all_x = self.train_x if is_train else self.test_x
        all_adjs = self.train_adjs if is_train else self.test_adjs
        all_indices = self.train_indices if is_train else self.test_indices

        # 随机采样 N_way 个类别
        all_classes = list(all_indices.keys())
        if len(all_classes) < N_way:
            raise ValueError(f"类别数不足: 仅有 {len(all_classes)} 类，要求 {N_way} 类")
        class_index = np.random.choice(all_classes, N_way, replace=False)

        support_indices = []
        query_indices = []
        append_count = 0

        # 为每个类别采样支持集和查询集
        for cls_idx in class_index:
            indices = list(all_indices[cls_idx])
            if len(indices) < K_shot + R_query and (is_train or test_start_idx is None):
                print(
                    f"警告: 类别 {cls_idx} 的样本不足 (需要 {K_shot + R_query}, 实际 {len(indices)})"
                )

            if is_train or test_start_idx is None:
                np.random.shuffle(indices)

            # 采样支持集
            cls_support = indices[: min(K_shot, len(indices))]
            if len(cls_support) < K_shot:
                # 不足时重复采样
                cls_support = (cls_support * (K_shot // len(cls_support) + 1))[:K_shot]
            support_indices.extend(cls_support)

            # 采样查询集(只在非测试模式或未指定test_start_idx时)
            if is_train or test_start_idx is None:
                remaining = indices[min(K_shot, len(indices)) :]
                if len(remaining) < R_query:
                    remaining = (
                        remaining * (R_query // len(remaining) + 1 if remaining else 1)
                    )[:R_query]
                query_indices.extend(remaining[:R_query])

        # 特殊处理测试查询集
        if not is_train and test_start_idx is not None:
            query_indices = []
            for i, cls_idx in enumerate(class_index):
                start = min(test_start_idx + i * R_query, len(self.total_test_indices))
                end = min(start + R_query, len(self.total_test_indices))
                current_indices = self.total_test_indices[start:end]

                while len(current_indices) < R_query:
                    if current_indices:
                        current_indices.append(current_indices[-1])
                    elif support_indices:
                        current_indices.append(support_indices[0])
                    else:
                        print(f"Warning: 无法填充类别 {cls_idx} 的查询集 - 数据不足")
                        break
                    append_count += 1
                query_indices.extend(current_indices)

        if not query_indices:
            print("警告: 查询集为空，将使用支持集的一部分作为查询集")
            split_point = len(support_indices) // 2
            query_indices = support_indices[:split_point]
            support_indices = support_indices[split_point:]

        support_x = all_x[support_indices]
        support_adj = all_adjs[support_indices]
        query_x = all_x[query_indices]
        query_adj = all_adjs[query_indices]

        support_samples_per_class = len(support_indices) // N_way
        query_samples_per_class = len(query_indices) // N_way

        support_label = torch.zeros(len(support_indices), dtype=torch.long)
        query_label = torch.zeros(len(query_indices), dtype=torch.long)

        for i in range(N_way):
            s_start, s_end = i * support_samples_per_class, (i + 1) * support_samples_per_class
            q_start, q_end = i * query_samples_per_class, (i + 1) * query_samples_per_class

            if s_start < len(support_label):
                support_label[s_start : min(s_end, len(support_label))] = i

            if q_start < len(query_label):
                query_label[q_start : min(q_end, len(query_label))] = i

        return {
            "support_set": {"x": support_x, "adj": support_adj, "label": support_label},
            "query_set": {"x": query_x, "adj": query_adj, "label": query_label},
            "append_count": append_count,
        }
