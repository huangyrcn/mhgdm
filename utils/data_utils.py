# --- 高效数据加载与采样（参考SMART的标签处理策略） ---
import os
import pathlib
import pickle
import networkx as nx
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils.graph_utils import graphs_to_tensor
import json
from collections import defaultdict
from numpy.random import RandomState
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings


# --- 优化后的数据文件读取与预处理 ---
def _process_single_graph(
    graph_data, use_degree_as_tag, original_node_tag_map, tag_to_idx_map, feature_dimension
):
    """处理单个图的辅助函数，支持并行处理"""
    graph_item = graph_data
    nx_g = graph_item["nx_graph"]
    nx_g.graph["label"] = graph_item["label"]

    if use_degree_as_tag:
        node_tags_for_features = [d for _, d in nx_g.degree()]
    else:
        node_tags_for_features = graph_item["raw_tags"]

    for node_id, node_obj in nx_g.nodes(data=True):
        tag_val = node_tags_for_features[node_id]
        one_hot_vec = np.zeros(feature_dimension, dtype=np.float32)
        if feature_dimension > 0 and tag_val in tag_to_idx_map:
            one_hot_vec[tag_to_idx_map[tag_val]] = 1.0
        elif feature_dimension == 1 and not tag_to_idx_map:
            one_hot_vec[0] = 0.0
        nx_g.nodes[node_id]["feature"] = one_hot_vec

    return nx_g


def load_from_file_optimized(data_config, use_degree_as_tag):
    """
    优化版本的数据加载函数，参考SMART框架保留原始标签
    """
    dataset_name = data_config.name
    base_dir = pathlib.Path("./datasets") / dataset_name
    file_path = base_dir / f"{dataset_name}.txt"
    save_file_path = base_dir / f"{dataset_name}_processed.pkl"  # 使用与原版相同的文件名
    force_reload = getattr(data_config, "force_reload_data", False)

    if not force_reload and save_file_path.exists():
        print(f"Loading optimized processed data from: {save_file_path}")
        with open(save_file_path, "rb") as f:
            data = pickle.load(f)
        # 检查文件格式，如果是原版格式，转换为优化版格式
        if "all_nx_graphs" in data:
            # 原版格式，需要转换为GraphData格式
            all_nx_graphs = data["all_nx_graphs"]
            all_graph_data = []
            for nx_g in all_nx_graphs:
                original_label = nx_g.graph["label"]
                graph_data = GraphData(nx_g, original_label)
                all_graph_data.append(graph_data)
            return all_graph_data, data["tagset"], data["max_node_num"], data["max_feat_dim"]
        else:
            # 优化版格式
            return (
                data["all_graph_data"],
                data["tagset"],
                data["max_node_num"],
                data["max_feat_dim"],
            )

    print(
        f"Loading and processing data from: {file_path} (optimized version with label preservation)"
    )

    # Phase 1: Fast file reading - 保留原始标签
    temp_graph_data_list = []
    original_node_tag_map = {}
    max_nodes_observed = 0
    tags_for_onehot_basis = set()

    start_time = time.time()
    with open(file_path, "r") as f:
        lines = f.readlines()

    num_graphs_in_file = int(lines[0].strip())
    temp_graph_data_list = [None] * num_graphs_in_file

    line_idx = 1
    for graph_idx in range(num_graphs_in_file):
        meta_line = lines[line_idx].strip().split()
        num_nodes, original_g_label = int(meta_line[0]), int(meta_line[1])  # 保留原始标签
        line_idx += 1

        edges = []
        current_g_raw_tags = []
        max_nodes_observed = max(max_nodes_observed, num_nodes)

        for node_i in range(num_nodes):
            node_line_parts = lines[line_idx].strip().split()
            line_idx += 1

            original_node_tag = int(node_line_parts[0])
            if original_node_tag not in original_node_tag_map:
                original_node_tag_map[original_node_tag] = len(original_node_tag_map)
            mapped_tag = original_node_tag_map[original_node_tag]
            current_g_raw_tags.append(mapped_tag)

            if not use_degree_as_tag:
                tags_for_onehot_basis.add(mapped_tag)

            num_neighbors = int(node_line_parts[1])
            for k in range(num_neighbors):
                neighbor_id = int(node_line_parts[2 + k])
                if node_i < neighbor_id:
                    edges.append((node_i, neighbor_id))

        # 关键：保存原始标签，不进行任何映射
        temp_graph_data_list[graph_idx] = {
            "edges": edges,
            "num_nodes": num_nodes,
            "original_label": original_g_label,  # 保留原始标签
            "raw_tags": current_g_raw_tags,
        }

    print(f"Phase 1 (Graph loading with label preservation): {time.time() - start_time:.2f}s")

    # Phase 2: Degree calculation if needed
    if use_degree_as_tag:
        start_time = time.time()
        for graph_item in temp_graph_data_list:
            degree_count = [0] * graph_item["num_nodes"]
            for i, j in graph_item["edges"]:
                degree_count[i] += 1
                degree_count[j] += 1
            graph_item["raw_tags"] = degree_count
            tags_for_onehot_basis.update(degree_count)
        print(f"Phase 2 (Degree calculation): {time.time() - start_time:.2f}s")

    # Phase 3: Feature encoding
    final_tagset = sorted(list(tags_for_onehot_basis))
    tag_to_idx_map = {tag: i for i, tag in enumerate(final_tagset)}
    feature_dimension = len(final_tagset)
    if feature_dimension == 0 and num_graphs_in_file > 0:
        print("Warning: Feature dimension is 0. Defaulting to a 1-dim zero feature.")
        feature_dimension = 1

    # Phase 4: Build GraphData objects with NetworkX graphs
    start_time = time.time()

    def build_graph_data(graph_data_item):
        """构建GraphData对象，保留原始标签"""
        edges = graph_data_item["edges"]
        num_nodes = graph_data_item["num_nodes"]
        raw_tags = graph_data_item["raw_tags"]
        original_label = graph_data_item["original_label"]  # 使用原始标签

        # 构建NetworkX图
        nx_g = nx.Graph()
        nx_g.add_nodes_from(range(num_nodes))
        nx_g.add_edges_from(edges)

        # 添加节点特征
        for node_id in range(num_nodes):
            tag_val = raw_tags[node_id]
            one_hot_vec = np.zeros(feature_dimension, dtype=np.float32)
            if feature_dimension > 0 and tag_val in tag_to_idx_map:
                one_hot_vec[tag_to_idx_map[tag_val]] = 1.0
            elif feature_dimension == 1 and not tag_to_idx_map:
                one_hot_vec[0] = 0.0
            nx_g.nodes[node_id]["feature"] = one_hot_vec

        # 创建GraphData对象，保留原始标签
        return GraphData(nx_g, original_label)

    max_workers = min(mp.cpu_count(), 8)
    if len(temp_graph_data_list) > 100 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_graph_data_list = list(executor.map(build_graph_data, temp_graph_data_list))
    else:
        processed_graph_data_list = [build_graph_data(g) for g in temp_graph_data_list]

    print(
        f"Phase 4 (GraphData construction with label preservation): {time.time() - start_time:.2f}s"
    )

    # 保存处理后的数据
    data_to_save = {
        "all_graph_data": processed_graph_data_list,
        "tagset": final_tagset,
        "max_node_num": max_nodes_observed,
        "max_feat_dim": feature_dimension,
    }
    with open(save_file_path, "wb") as f:
        pickle.dump(data_to_save, f)

    return processed_graph_data_list, final_tagset, max_nodes_observed, feature_dimension


class MyDatasetOptimized:
    """
    优化版数据集类，专为Letter_high和Few-shot Learning设计
    包含原始标签保留和高效FSL任务采样
    """

    def __init__(self, data_config, fsl_task_config=None):
        self.config = data_config
        self.fsl_config = fsl_task_config
        self.dataset_name = data_config.name

        print(f"Initializing MyDataset for {self.dataset_name}...")

        # 数据加载
        self._load_data()

        # 转换为张量格式
        self._convert_to_tensors()

        # FSL索引构建
        if fsl_task_config:
            self._build_fsl_indices()

        self._print_dataset_summary()

    def _load_data(self):
        """加载数据，优先使用缓存"""
        degree_as_tag = getattr(self.config, "degree_as_tag", False)
        self.all_graph_data, self.tagset, self.max_node_num, self.max_feat_dim = (
            load_from_file_optimized(self.config, degree_as_tag)
        )

        # 分析标签分布
        label_counts = defaultdict(int)
        for graph_data in self.all_graph_data:
            original_label = graph_data.get_label("original")
            label_counts[original_label] += 1

        # 确定数据分割策略
        if hasattr(self.config, "test_class_num") and hasattr(self.config, "train_class_num"):
            # 显式指定训练和测试类别数
            self.test_class_num = self.config.test_class_num
            self.train_class_num = self.config.train_class_num
        else:
            # 根据test_split比例自动分割
            test_split = getattr(self.config, "test_split", 0.2)
            total_classes = len(label_counts)
            self.test_class_num = max(1, int(total_classes * test_split))
            self.train_class_num = total_classes - self.test_class_num

        # 根据样本数量排序，确保测试类有足够样本
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        # Letter_high特定的类别分配（如果可用）
        if self.dataset_name == "Letter_high" and len(sorted_labels) >= 15:
            # 修复：使用数据驱动的分割而不是硬编码的预定义分割
            # 原来的硬编码分割有问题，导致测试集为空
            # train_classes = [1, 9, 10, 2, 0, 3, 14, 5, 12, 13, 7]  # 11个训练类
            # test_classes = [4, 6, 11, 8]  # 4个测试类

            # 新的数据驱动分割：基于样本数量选择最优分割
            # 选择样本数最多的前11个类别作为训练，后4个作为测试
            all_labels_sorted = [label for label, _ in sorted_labels]
            train_classes = all_labels_sorted[: self.train_class_num]  # 前11个
            test_classes = all_labels_sorted[
                self.train_class_num : self.train_class_num + self.test_class_num
            ]  # 后4个

            print(f"Letter_high 数据驱动分割:")
            print(f"  训练类别: {train_classes}")
            print(f"  测试类别: {test_classes}")

            # 验证类别是否存在（这里应该都存在，因为是从sorted_labels中选择的）
            available_labels = set(label_counts.keys())
            valid_train_classes = [c for c in train_classes if c in available_labels]
            valid_test_classes = [c for c in test_classes if c in available_labels]

            if (
                len(valid_train_classes) >= self.train_class_num
                and len(valid_test_classes) >= self.test_class_num
            ):
                self.train_classes = valid_train_classes[: self.train_class_num]
                self.test_classes = valid_test_classes[: self.test_class_num]
            else:
                # 如果数据驱动分割失败，使用自动分配（这种情况应该不会发生）
                print("警告：数据驱动分割失败，使用自动分配")
                self.test_classes = [label for label, _ in sorted_labels[: self.test_class_num]]
                self.train_classes = [
                    label
                    for label, _ in sorted_labels[
                        self.test_class_num : self.test_class_num + self.train_class_num
                    ]
                ]
        else:
            # 其他数据集的自动分配
            self.test_classes = [label for label, _ in sorted_labels[: self.test_class_num]]
            self.train_classes = [
                label
                for label, _ in sorted_labels[
                    self.test_class_num : self.test_class_num + self.train_class_num
                ]
            ]

        # 为图数据分配训练/测试标签
        train_label_map = {
            original_label: idx for idx, original_label in enumerate(self.train_classes)
        }
        test_label_map = {
            original_label: idx for idx, original_label in enumerate(self.test_classes)
        }

        self.train_graphs = []
        self.test_graphs = []

        for graph_data in self.all_graph_data:
            original_label = graph_data.get_label("original")
            if original_label in train_label_map:
                graph_data.set_train_split(train_label_map[original_label])
                self.train_graphs.append(graph_data)
            elif original_label in test_label_map:
                graph_data.set_test_split(test_label_map[original_label])
                self.test_graphs.append(graph_data)

    def _convert_to_tensors(self):
        """将图数据转换为张量格式"""
        print("Converting graphs to tensors...")

        def graphs_to_tensor_batch(graph_data_list):
            max_node_num = self.config.max_node_num
            max_feat_num = self.config.max_feat_num
            adj_tensor, x_tensor = graphs_to_tensor(
                [gd.nx_graph for gd in graph_data_list], max_node_num, max_feat_num
            )
            labels = torch.tensor(
                [
                    (
                        gd.get_label("train_split")
                        if gd.get_label("train_split") is not None
                        else gd.get_label("test_split")
                    )
                    for gd in graph_data_list
                ],
                dtype=torch.long,
            )
            return x_tensor, adj_tensor, labels

        self.train_x, self.train_adj, self.train_labels = graphs_to_tensor_batch(self.train_graphs)
        self.test_x, self.test_adj, self.test_labels = graphs_to_tensor_batch(self.test_graphs)

        print(f"✓ Tensor conversion completed:")
        print(f"  Train tensors: x{self.train_x.shape}, adj{self.train_adj.shape}")
        print(f"  Test tensors: x{self.test_x.shape}, adj{self.test_adj.shape}")

    def _build_fsl_indices(self):
        """构建FSL任务的索引映射"""
        if not self.fsl_config:
            return

        # 训练集索引（按类别）
        self.train_indices_by_class = defaultdict(list)
        for idx, graph_data in enumerate(self.train_graphs):
            class_id = graph_data.get_label("train_split")
            self.train_indices_by_class[class_id].append(idx)

        # 测试集索引（按类别）
        self.test_indices_by_class = defaultdict(list)
        for idx, graph_data in enumerate(self.test_graphs):
            class_id = graph_data.get_label("test_split")
            self.test_indices_by_class[class_id].append(idx)

        # 调试信息：打印FSL索引构建结果
        print(f"FSL索引构建结果:")
        print(f"  训练集类别分布: {dict(self.train_indices_by_class)}")
        print(f"  测试集类别分布: {dict(self.test_indices_by_class)}")

        # 更详细的调试：检查测试图的标签
        if len(self.test_graphs) > 0:
            print(f"  测试图详细信息（前5个）:")
            for i, graph_data in enumerate(self.test_graphs[:5]):
                print(
                    f"    Graph {i}: original_label={graph_data.get_label('original')}, test_split_label={graph_data.get_label('test_split')}"
                )
        else:
            print(f"  ⚠️ 警告：没有测试图!")

        # 检查测试类别与实际分配的匹配
        assigned_test_classes = set(self.test_indices_by_class.keys())
        # 修复：期望的测试类别应该是重映射后的标签（0,1,2,3），而不是原始标签
        expected_test_classes = set(range(len(self.test_classes)))  # {0, 1, 2, 3}
        print(f"  期望测试类别: {expected_test_classes} (重映射后)")
        print(f"  实际分配类别: {assigned_test_classes}")
        print(f"  原始测试类别: {self.test_classes}")

        if assigned_test_classes != expected_test_classes:
            print(f"  ⚠️ 类别分配不匹配!")
            missing_classes = expected_test_classes - assigned_test_classes
            unexpected_classes = assigned_test_classes - expected_test_classes
            if missing_classes:
                print(f"     缺失类别: {missing_classes}")
            if unexpected_classes:
                print(f"     意外类别: {unexpected_classes}")
        else:
            print(f"  ✅ 类别分配正确!")

    def get_loaders(self):
        """返回训练和测试数据加载器"""
        batch_size = getattr(self.config, "batch_size", 64)

        train_dataset = TensorDataset(self.train_x, self.train_adj, self.train_labels)
        test_dataset = TensorDataset(self.test_x, self.test_adj, self.test_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        return train_loader, test_loader

    def _prepare_task_class_samples(self, all_indices_for_class_list, num_needed):
        """为任务准备类别样本"""
        selected_indices = []
        for class_indices in all_indices_for_class_list:
            if len(class_indices) >= num_needed:
                selected = np.random.choice(class_indices, num_needed, replace=False).tolist()
            else:
                selected = (
                    class_indices
                    + np.random.choice(
                        class_indices, num_needed - len(class_indices), replace=True
                    ).tolist()
                )
            selected_indices.extend(selected)
        return selected_indices

    def _sample_indices_for_task(
        self, indices_map_by_class, list_of_available_classes, N_way, K_shot, R_query
    ):
        """为FSL任务采样索引"""
        # 过滤掉样本数不足的类别
        valid_classes = [
            c for c in list_of_available_classes if len(indices_map_by_class.get(c, [])) > 0
        ]

        if len(valid_classes) < N_way:
            return None

        selected_classes = np.random.choice(valid_classes, N_way, replace=False)

        support_class_indices = []
        query_class_indices = []

        for class_id in selected_classes:
            class_indices = indices_map_by_class[class_id]
            total_needed = K_shot + R_query

            if len(class_indices) >= total_needed:
                selected_indices = np.random.choice(
                    class_indices, total_needed, replace=False
                ).tolist()
                support_indices = selected_indices[:K_shot]
                query_indices = selected_indices[K_shot:]
            else:
                # 处理样本不足的情况
                selected_indices = (
                    class_indices
                    + np.random.choice(
                        class_indices, total_needed - len(class_indices), replace=True
                    ).tolist()
                )
                support_indices = selected_indices[:K_shot]
                query_indices = selected_indices[K_shot : K_shot + R_query]

            support_class_indices.append(support_indices)
            query_class_indices.append(query_indices)

        return {
            "selected_classes": selected_classes,
            "support_indices": support_class_indices,
            "query_indices": query_class_indices,
        }

    def sample_one_task(self, is_train, N_way, K_shot, R_query, query_pool_start_index=None):
        """采样一个FSL任务"""
        if is_train:
            indices_map = self.train_indices_by_class
            # 训练集：使用重映射后的类别ID（0到len(train_classes)-1）
            available_classes = list(range(len(self.train_classes)))
            x_tensor, adj_tensor = self.train_x, self.train_adj
        else:
            indices_map = self.test_indices_by_class
            # 测试集：使用重映射后的类别ID（0到len(test_classes)-1）
            available_classes = list(range(len(self.test_classes)))
            x_tensor, adj_tensor = self.test_x, self.test_adj

        task_data = self._sample_indices_for_task(
            indices_map, available_classes, N_way, K_shot, R_query
        )

        if task_data is None:
            return None

        # 构建支持集
        support_indices = []
        support_labels = []
        for class_idx, class_support_indices in enumerate(task_data["support_indices"]):
            support_indices.extend(class_support_indices)
            support_labels.extend([class_idx] * len(class_support_indices))

        # 构建查询集
        query_indices = []
        query_labels = []
        for class_idx, class_query_indices in enumerate(task_data["query_indices"]):
            query_indices.extend(class_query_indices)
            query_labels.extend([class_idx] * len(class_query_indices))

        # 提取张量数据
        support_x = x_tensor[support_indices]
        support_adj = adj_tensor[support_indices]
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long)

        query_x = x_tensor[query_indices]
        query_adj = adj_tensor[query_indices]
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)

        return {
            "support_set": {
                "x": support_x,
                "adj": support_adj,
                "label": support_labels_tensor,
            },
            "query_set": {
                "x": query_x,
                "adj": query_adj,
                "label": query_labels_tensor,
            },
            "N_way": N_way,
            "K_shot": K_shot,
            "R_query": R_query,
            "selected_classes": task_data["selected_classes"],
        }

    def _print_dataset_summary(self):
        """打印数据集摘要"""
        print(f"✓ Dataset loaded:")
        print(f"  Train graphs: {len(self.train_graphs)}")
        print(f"  Test graphs: {len(self.test_graphs)}")
        print(f"  Train classes: {self.train_class_num}")
        print(f"  Test classes: {self.test_class_num}")
        print(f"  Max nodes: {self.max_node_num}")
        print(f"  Max features: {self.max_feat_dim}")


def load_from_file(data_config, use_degree_as_tag):
    """向后兼容函数"""
    warnings.warn(
        "load_from_file is deprecated, use load_from_file_optimized instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_from_file_optimized(data_config, use_degree_as_tag)


def load_data(config, get_graph_list=False):
    """向后兼容的数据加载函数"""
    fsl_task_config = getattr(config, "fsl_task", None)
    dataset = MyDatasetOptimized(config.data, fsl_task_config)

    if get_graph_list:
        return dataset, dataset.all_graph_data
    else:
        return dataset


class DataLoadingProfiler:
    """数据加载性能分析工具"""

    @staticmethod
    def profile_data_loading(data_config, fsl_task_config=None, num_runs=3):
        """分析数据加载性能"""
        times = []

        for i in range(num_runs):
            start_time = time.time()
            dataset_old = MyDatasetOptimized(data_config, fsl_task_config)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Run {i+1}: {times[-1]:.2f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\nAverage loading time: {avg_time:.2f}s ± {std_time:.2f}s")

        # 内存使用分析
        print(f"Train graphs: {len(dataset_old.train_graphs)}")
        print(f"Test graphs: {len(dataset_old.test_graphs)}")

        # 分析数据质量
        train_sizes = [len(indices) for indices in dataset_old.train_indices_by_class.values()]
        test_sizes = [len(indices) for indices in dataset_old.test_indices_by_class.values()]

        print(
            f"Train class sizes: min={min(train_sizes)}, max={max(train_sizes)}, avg={np.mean(train_sizes):.1f}"
        )
        print(
            f"Test class sizes: min={min(test_sizes)}, max={max(test_sizes)}, avg={np.mean(test_sizes):.1f}"
        )

        # 采样任务测试
        if fsl_task_config:
            start_time = time.time()
            for _ in range(10):
                task = dataset_old.sample_one_task(
                    is_train=False,
                    N_way=fsl_task_config.N_way,
                    K_shot=fsl_task_config.K_shot,
                    R_query=fsl_task_config.R_query,
                )
            end_time = time.time()
            print(f"10 FSL task sampling time: {end_time - start_time:.2f}s")

        return dataset_old


class GraphData:
    """
    图数据包装类，支持原始标签和训练/测试标签
    """

    def __init__(self, nx_graph, original_label):
        self.nx_graph = nx_graph
        self.original_label = original_label
        self.train_split_label = None
        self.test_split_label = None

    def __repr__(self):
        return (
            f"GraphData(nodes={self.nx_graph.number_of_nodes()}, "
            f"edges={self.nx_graph.number_of_edges()}, "
            f"original_label={self.original_label})"
        )

    def get_label(self, mode="original"):
        """
        获取标签
        mode: "original", "train_split", "test_split"
        """
        if mode == "original":
            return self.original_label
        elif mode == "train_split":
            return self.train_split_label
        elif mode == "test_split":
            return self.test_split_label
        else:
            raise ValueError(f"Unknown label mode: {mode}")

    def set_train_split(self, remapped_label):
        """设置训练分割标签"""
        self.train_split_label = remapped_label

    def set_test_split(self, remapped_label):
        """设置测试分割标签"""
        self.test_split_label = remapped_label


# 为了向后兼容，创建MyDataset别名
MyDataset = MyDatasetOptimized
