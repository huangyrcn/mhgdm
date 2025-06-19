"""
高效数据工具 - 动态批处理，消除内存浪费
"""

import os
import pathlib
import pickle
import networkx as nx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from collections import defaultdict
from numpy.random import RandomState
import warnings


def graphs_to_tensor_robust(graph_list, max_node_num, max_feat_num):
    """
    健壮的图转张量函数 - 支持tag和feature属性
    """
    adjs_list = []
    x_list = []

    for g in graph_list:
        assert isinstance(g, nx.Graph)

        # 获取节点列表（按顺序）
        node_list = list(g.nodes())

        # 构建邻接矩阵
        adj = nx.to_numpy_array(g, nodelist=node_list)

        # 填充邻接矩阵
        adj = _pad_adjs(adj, max_node_num)
        adjs_list.append(adj)

        # 构建特征矩阵
        feature_list = []
        for node in node_list:
            node_data = g.nodes[node]

            if "feature" in node_data:
                # 使用显式特征
                features = node_data["feature"]
                if isinstance(features, (list, np.ndarray)):
                    feature_list.append(np.array(features, dtype=np.float32))
                else:
                    feature_list.append(np.array([features], dtype=np.float32))
            elif "tag" in node_data:
                # 使用one-hot编码的度数标签
                tag = node_data["tag"]
                one_hot = np.zeros(max_feat_num, dtype=np.float32)
                if tag < max_feat_num:
                    one_hot[tag] = 1.0
                feature_list.append(one_hot)
            else:
                # 默认使用度数作为特征
                degree = g.degree(node)
                one_hot = np.zeros(max_feat_num, dtype=np.float32)
                if degree < max_feat_num:
                    one_hot[degree] = 1.0
                feature_list.append(one_hot)

        # 转换为数组并填充
        if feature_list:
            x = np.stack(feature_list, axis=0)
            x = _pad_features(x, max_node_num, max_feat_num)
            x_list.append(x)

    adjs_tensor = torch.tensor(np.asarray(adjs_list), dtype=torch.float32)
    x_tensor = torch.tensor(np.asarray(x_list), dtype=torch.float32)

    return adjs_tensor, x_tensor


def _pad_adjs(adj, node_number):
    """填充邻接矩阵到指定大小"""
    a = adj
    ori_len = a.shape[0]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f"ori_len {ori_len} > node_number {node_number}")
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def _pad_features(x, node_number, feature_dim):
    """填充特征矩阵到指定大小"""
    n_nodes, feat_dim = x.shape
    if n_nodes > node_number:
        raise ValueError(f"n_nodes {n_nodes} > node_number {node_number}")
    if feat_dim > feature_dim:
        raise ValueError(f"feat_dim {feat_dim} > feature_dim {feature_dim}")

    padded_x = np.zeros((node_number, feature_dim), dtype=x.dtype)
    padded_x[:n_nodes, :feat_dim] = x
    return padded_x


def load_from_file(data_config, use_degree_as_tag):
    """
    数据加载函数，增强了错误处理和性能优化
    """
    dataset_name = data_config.name
    base_dir = pathlib.Path("./datasets") / dataset_name
    file_path = base_dir / f"{dataset_name}.txt"
    save_file_path = base_dir / f"{dataset_name}_processed.pkl"
    force_reload = getattr(data_config, "force_reload_data", False)

    if not force_reload and save_file_path.exists():
        print(f"Loading processed data from: {save_file_path}")
        try:
            with open(save_file_path, "rb") as f:
                data = pickle.load(f)
            return data["all_nx_graphs"], data["tagset"], data["max_node_num"], data["max_feat_dim"]
        except (ModuleNotFoundError, AttributeError, ImportError) as e:
            if "numpy._core" in str(e) or "numpy.core" in str(e):
                print(f"⚠️  Numpy version compatibility issue detected: {e}")
                print("🔧 Regenerating processed data file with current numpy version...")
                force_reload = True
            else:
                raise e
        except Exception as e:
            print(f"⚠️  Error loading processed data: {e}")
            print("🔧 Regenerating processed data file...")
            force_reload = True

    if force_reload or not save_file_path.exists():
        print(f"Processing raw data from: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        all_nx_graphs = []
        tagset = set()

        try:
            with open(file_path, "r") as f:
                n_graphs = int(f.readline().strip())
                print(f"Loading {n_graphs} graphs...")

                for i in range(n_graphs):
                    line = f.readline().strip().split()
                    n_nodes, graph_label = int(line[0]), int(line[1])

                    g = nx.Graph()
                    g.graph["label"] = graph_label

                    for j in range(n_nodes):
                        node_line = f.readline().strip().split()
                        node_id = int(node_line[0])
                        n_neighbors = int(node_line[1])

                        # 处理节点特征
                        feature_start_idx = 2 + n_neighbors
                        if len(node_line) > feature_start_idx:
                            # 有显式特征
                            features = [float(x) for x in node_line[feature_start_idx:]]
                            g.add_node(node_id, features=features)
                        else:
                            # 使用度数作为特征
                            if use_degree_as_tag:
                                g.add_node(node_id, tag=n_neighbors)
                                tagset.add(n_neighbors)
                            else:
                                g.add_node(node_id, features=[1.0])

                        # 添加边
                        for k in range(2, 2 + n_neighbors):
                            neighbor = int(node_line[k])
                            g.add_edge(node_id, neighbor)

                    all_nx_graphs.append(g)

                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i + 1}/{n_graphs} graphs")

        except Exception as e:
            print(f"Error reading dataset file: {e}")
            raise

        # 计算统计信息
        max_node_num = max(len(g.nodes()) for g in all_nx_graphs)

        # 计算特征维度
        max_feat_dim = 1  # 默认值
        for g in all_nx_graphs:
            for node in g.nodes():
                node_data = g.nodes[node]
                if "feature" in node_data:
                    features = node_data["feature"]
                    if isinstance(features, (list, np.ndarray)):
                        feat_dim = len(features)
                    else:
                        feat_dim = 1
                    max_feat_dim = max(max_feat_dim, feat_dim)
                    break

        if use_degree_as_tag and tagset:
            max_feat_dim = len(tagset)

        print(f"Dataset statistics:")
        print(f"  Total graphs: {len(all_nx_graphs)}")
        print(f"  Max nodes: {max_node_num}")
        print(f"  Max feature dim: {max_feat_dim}")
        print(f"  Tagset size: {len(tagset)}")

        # 保存处理后的数据
        try:
            save_data = {
                "all_nx_graphs": all_nx_graphs,
                "tagset": tagset,
                "max_node_num": max_node_num,
                "max_feat_dim": max_feat_dim,
            }
            with open(save_file_path, "wb") as f:
                pickle.dump(save_data, f)
            print(f"✓ Processed data saved to: {save_file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save processed data: {e}")

        return all_nx_graphs, tagset, max_node_num, max_feat_dim


class MyDataset:
    """
    高效数据管理器 - 预处理张量化模式
    在初始化时一次性将所有NetworkX图转换为张量，然后使用TensorDataset
    提供三个主要接口：
    1. get_loaders() - 获取DataLoader（使用预处理张量）
    2. get_graph_lists() - 获取原始图列表
    3. sample_one_task() - 获取FSL任务
    """

    def __init__(self, data_config, fsl_task_config=None):
        self.data_config = data_config
        self.fsl_task_config = fsl_task_config
        use_degree_as_tag = getattr(data_config, "degree_as_tag", True)

        print(f"Initializing MyDataset for {data_config.name}...")
        all_nx_graphs, self.tagset, max_nodes_data, feat_dim_data = load_from_file(
            data_config, use_degree_as_tag
        )

        self.max_node_num = getattr(data_config, "max_node_num", max_nodes_data)
        # 兼容不同的特征维度字段名
        if hasattr(data_config, "max_feat_dim"):
            self.max_feat_num = getattr(data_config, "max_feat_dim")
        elif hasattr(data_config, "max_feat_num"):
            self.max_feat_num = getattr(data_config, "max_feat_num")
        else:
            self.max_feat_num = len(self.tagset) if len(self.tagset) > 0 else feat_dim_data

        # 数据集划分
        split_file = pathlib.Path(f"./datasets/{data_config.name}/train_test_classes.json")
        with open(split_file, "r") as f:
            class_splits = json.load(f)
            self.train_original_classes_set = set(map(int, class_splits["train"]))
            self.test_original_classes_set = set(map(int, class_splits["test"]))

        # 按训练集和测试集分割数据
        self.train_nx_graphs = []
        train_orig_labels_list = []
        self.test_nx_graphs = []
        test_orig_labels_list = []

        for nx_g in all_nx_graphs:
            original_label = nx_g.graph["label"]
            if original_label in self.train_original_classes_set:
                self.train_nx_graphs.append(nx_g)
                train_orig_labels_list.append(original_label)
            elif original_label in self.test_original_classes_set:
                self.test_nx_graphs.append(nx_g)
                test_orig_labels_list.append(original_label)

        # 标签重映射
        unique_train_sorted = sorted(list(self.train_original_classes_set))
        train_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_train_sorted)}
        self.train_labels_remapped = torch.LongTensor(
            [train_remapper[l] for l in train_orig_labels_list]
        )
        self.num_train_classes_remapped = len(unique_train_sorted)

        unique_test_sorted = sorted(list(self.test_original_classes_set))
        test_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_test_sorted)}
        self.test_labels_remapped = torch.LongTensor(
            [test_remapper[l] for l in test_orig_labels_list]
        )
        self.num_test_classes_remapped = len(unique_test_sorted)

        # 设置图列表接口（向后兼容）
        self.train_graphs = self.train_nx_graphs
        self.test_graphs = self.test_nx_graphs

        print(f"✓ Dataset loaded:")
        print(f"  Train graphs: {len(self.train_nx_graphs)}")
        print(f"  Test graphs: {len(self.test_nx_graphs)}")
        print(f"  Train classes: {self.num_train_classes_remapped}")
        print(f"  Test classes: {self.num_test_classes_remapped}")
        print(f"  Max nodes: {self.max_node_num}")
        print(f"  Max features: {self.max_feat_num}")

        # **关键改动：一次性预处理所有图为张量**
        print("Converting graphs to tensors...")
        # 注意：graphs_to_tensor_robust返回(adjs_tensor, x_tensor)
        train_adjs, train_x = graphs_to_tensor_robust(
            self.train_nx_graphs, self.max_node_num, self.max_feat_num
        )
        test_adjs, test_x = graphs_to_tensor_robust(
            self.test_nx_graphs, self.max_node_num, self.max_feat_num
        )

        # 正确存储：x是特征，adj是邻接矩阵
        self.train_x = train_x
        self.train_adj = train_adjs
        self.test_x = test_x
        self.test_adj = test_adjs

        print(f"✓ Tensor conversion completed:")
        print(f"  Train tensors: x{self.train_x.shape}, adj{self.train_adj.shape}")
        print(f"  Test tensors: x{self.test_x.shape}, adj{self.test_adj.shape}")

        # FSL索引构建
        if self.fsl_task_config is not None:
            self._build_fsl_indices()

    def _build_fsl_indices(self):
        """构建FSL任务所需的索引"""
        # 构建类别到样本索引的映射
        self.train_indices_by_class = defaultdict(list)
        for i, remapped_label in enumerate(self.train_labels_remapped.tolist()):
            self.train_indices_by_class[remapped_label].append(i)

        self.test_indices_by_class = defaultdict(list)
        for i, remapped_label in enumerate(self.test_labels_remapped.tolist()):
            self.test_indices_by_class[remapped_label].append(i)

        # 确定性测试采样支持集与查询池
        rng = np.random.RandomState(42)
        K_shot = getattr(self.fsl_task_config, "K_shot", 1)
        self.deterministic_test_support_indices_by_class = {}
        query_pool = []
        for cls, indices in self.test_indices_by_class.items():
            idxs = list(indices)
            rng.shuffle(idxs)
            self.deterministic_test_support_indices_by_class[cls] = idxs[:K_shot]
            query_pool.extend(idxs[K_shot:])
        rng.shuffle(query_pool)
        self.deterministic_test_query_pool_indices = query_pool

    # =============================================================================
    # 三大主要接口
    # =============================================================================

    def get_loaders(self):
        """
        接口1：获取DataLoader - 使用预处理张量和TensorDataset
        """
        batch_size = getattr(self.data_config, "batch_size", 32)
        num_workers = getattr(self.data_config, "num_workers", 0)

        # **关键改动：使用TensorDataset，不再使用动态collate**
        # 注意顺序：与load_batch函数期望的(x, adj, labels)保持一致
        train_dataset = TensorDataset(self.train_x, self.train_adj, self.train_labels_remapped)
        test_dataset = TensorDataset(self.test_x, self.test_adj, self.test_labels_remapped)

        # **关键改动：不传collate_fn，使用默认collate**
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # 可以开启pin_memory提升性能
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def get_graph_lists(self):
        """
        接口2：获取原始图列表
        """
        return self.train_nx_graphs, self.test_nx_graphs

    def sample_one_task(self, is_train, N_way, K_shot, R_query, query_pool_start_index=None):
        """
        接口3：采样FSL任务 - 直接从预处理的张量中采样
        """
        if self.fsl_task_config is None:
            raise RuntimeError("FSL task sampling requires fsl_task_config")

        # 选择数据源
        if is_train:
            indices_by_class = self.train_indices_by_class
            x_tensor = self.train_x
            adj_tensor = self.train_adj
            labels_tensor = self.train_labels_remapped
            num_classes = self.num_train_classes_remapped
        else:
            indices_by_class = self.test_indices_by_class
            x_tensor = self.test_x
            adj_tensor = self.test_adj
            labels_tensor = self.test_labels_remapped
            num_classes = self.num_test_classes_remapped

        # 检查可用类别数
        available_classes = [
            cls for cls, indices in indices_by_class.items() if len(indices) >= K_shot
        ]
        if len(available_classes) < N_way:
            print(f"Warning: Only {len(available_classes)} classes available, but N_way={N_way}")
            N_way = len(available_classes)

        # 随机选择N_way个类别
        rng = RandomState(None)  # 每次调用都使用新的随机状态
        selected_classes = rng.choice(available_classes, N_way, replace=False)

        # 采样支持集
        support_indices = []
        support_labels = []
        used_support_indices = set()  # 跟踪已使用的支持集索引

        for i, cls in enumerate(selected_classes):
            class_indices = indices_by_class[cls]
            selected_indices = rng.choice(class_indices, K_shot, replace=False)
            used_support_indices.update(selected_indices)  # 记录支持集索引

            for idx in selected_indices:
                support_indices.append(idx)
                support_labels.append(i)  # 重映射为0, 1, 2, ...

        # 采样查询集
        query_indices = []
        query_labels = []

        if is_train:
            # 训练时：从每个选中类别的剩余样本中随机采样
            for i, cls in enumerate(selected_classes):
                class_indices = indices_by_class[cls]
                # 排除已用作支持集的样本
                available_indices = [
                    idx for idx in class_indices if idx not in used_support_indices
                ]

                if len(available_indices) >= R_query:
                    selected_query_indices = rng.choice(available_indices, R_query, replace=False)
                    for idx in selected_query_indices:
                        query_indices.append(idx)
                        query_labels.append(i)  # 使用任务内的类别标签 (0, 1, 2, ...)
        else:
            # 测试时：使用预定义的查询池，但需要正确映射标签
            if query_pool_start_index is None:
                query_pool_start_index = 0

            query_pool_indices = self.deterministic_test_query_pool_indices
            total_query_needed = N_way * R_query
            end_index = min(query_pool_start_index + total_query_needed, len(query_pool_indices))

            if end_index - query_pool_start_index < total_query_needed:
                warnings.warn(
                    f"Not enough samples in query pool. "
                    f"Requested: {total_query_needed}, Available: {end_index - query_pool_start_index}"
                )

            selected_query_pool = query_pool_indices[query_pool_start_index:end_index]

            # **关键修复：根据样本的实际类别映射标签，而不是按顺序分配**
            for idx in selected_query_pool:
                # 获取样本的原始类别
                sample_original_class = labels_tensor[idx].item()

                # 检查这个类别是否在当前任务的selected_classes中
                if sample_original_class in selected_classes:
                    # 找到在selected_classes中的位置，作为任务内标签
                    task_label = list(selected_classes).index(sample_original_class)
                    query_indices.append(idx)
                    query_labels.append(task_label)

                    # 如果已经收集足够的查询样本，就停止
                    if len(query_labels) >= total_query_needed:
                        break

        # **关键改动：直接从预处理的张量中索引，不再需要实时转换图**
        support_x = x_tensor[support_indices]
        support_adj = adj_tensor[support_indices]
        support_labels_tensor = torch.LongTensor(support_labels)

        query_x = x_tensor[query_indices] if query_indices else torch.empty(0, *x_tensor.shape[1:])
        query_adj = (
            adj_tensor[query_indices] if query_indices else torch.empty(0, *adj_tensor.shape[1:])
        )
        query_labels_tensor = (
            torch.LongTensor(query_labels) if query_labels else torch.empty(0, dtype=torch.long)
        )

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
        }


# --- 向后兼容性别名 ---
DataManager = MyDataset


def load_data(config, get_graph_list=False):
    """
    向后兼容的数据加载函数
    现在使用预处理张量化的 MyDataset
    """
    dataset = MyDataset(config.data, getattr(config, "fsl_task", None))

    if get_graph_list:
        return dataset.get_graph_lists()
    else:
        return dataset.get_loaders()


# 数据加载别名（向后兼容）
load_dataset = load_data

# --- 向后兼容性说明 ---
"""
重要更新：MyDataset 已升级为预处理张量化模式

主要变化：
1. ✓ 删除了动态 collate 功能（smart_collate_fn）
2. ✓ 删除了 GraphDataset 类
3. ✓ 在初始化时一次性预处理所有图为张量
4. ✓ 使用 TensorDataset 替代动态数据集
5. ✓ DataLoader 使用默认 collate，不传入 collate_fn
6. ✓ 保持 (x, adj, label) 三元输出顺序

性能优势：
- 内存预分配，避免动态计算批次最大尺寸
- 消除实时图转换开销
- 支持 pin_memory 和多进程加载
- 更好的数据局部性和缓存友好性

接口兼容性：
- get_loaders() 接口保持不变
- load_batch() 函数无需修改
- 模型前向传播接口保持不变
- FSL 任务采样性能大幅提升（直接张量索引）
"""
