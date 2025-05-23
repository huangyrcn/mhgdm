

# --- 新实现：更简洁健壮的数据加载与采样 ---
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

# --- 数据文件读取与预处理 ---
def load_from_file(config, use_degree_as_tag):
    dataset_name = config.data.name
    base_dir = pathlib.Path("./datasets") / dataset_name
    file_path = base_dir / f"{dataset_name}.txt"
    save_file_path = base_dir / f"{dataset_name}_processed.pkl"
    force_reload = getattr(config.data, 'force_reload_data', False)

    if not force_reload and save_file_path.exists():
        print(f"Loading processed data from: {save_file_path}")
        with open(save_file_path, "rb") as f:
            data = pickle.load(f)
        return data["all_nx_graphs"], data["tagset"], \
               data["max_node_num"], data["max_feat_dim"]

    print(f"Loading and processing data from: {file_path}")
    temp_graph_data_list = []
    original_node_tag_map = {} 
    max_nodes_observed = 0
    tags_for_onehot_basis = set()

    with open(file_path, "r") as f:
        num_graphs_in_file = int(f.readline().strip())
        for _ in range(num_graphs_in_file):
            meta_line = f.readline().strip().split()
            num_nodes, original_g_label = int(meta_line[0]), int(meta_line[1])
            current_g = nx.Graph()
            current_g_raw_tags = []
            max_nodes_observed = max(max_nodes_observed, num_nodes)
            for node_i in range(num_nodes):
                current_g.add_node(node_i)
                node_line_parts = f.readline().strip().split()
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
                    current_g.add_edge(node_i, neighbor_id)
            temp_graph_data_list.append({
                "nx_graph": current_g, "label": original_g_label, "raw_tags": current_g_raw_tags
            })

    if use_degree_as_tag:
        tags_for_onehot_basis = set()
        for graph_item in temp_graph_data_list:
            nx_g = graph_item["nx_graph"]
            degrees = [d for _, d in nx_g.degree()]
            graph_item["raw_tags"] = degrees
            for deg in degrees:
                tags_for_onehot_basis.add(deg)

    final_tagset = sorted(list(tags_for_onehot_basis))
    tag_to_idx_map = {tag: i for i, tag in enumerate(final_tagset)}
    feature_dimension = len(final_tagset)
    if feature_dimension == 0 and num_graphs_in_file > 0:
        print("Warning: Feature dimension is 0. Defaulting to a 1-dim zero feature.")
        feature_dimension = 1 

    processed_nx_graphs = []
    for graph_item in temp_graph_data_list:
        nx_g = graph_item["nx_graph"]
        nx_g.graph['label'] = graph_item["label"]
        node_tags_for_features = graph_item["raw_tags"]
        for node_id, node_obj in nx_g.nodes(data=True):
            tag_val = node_tags_for_features[node_id]
            one_hot_vec = np.zeros(feature_dimension, dtype=np.float32)
            if feature_dimension > 0 and tag_val in tag_to_idx_map :
                 one_hot_vec[tag_to_idx_map[tag_val]] = 1.0
            elif feature_dimension == 1 and not tag_to_idx_map:
                one_hot_vec[0] = 0.0
            nx_g.nodes[node_id]['feature'] = one_hot_vec
        processed_nx_graphs.append(nx_g)
    data_to_save = {
        "all_nx_graphs": processed_nx_graphs, "tagset": final_tagset, 
        "max_node_num": max_nodes_observed, "max_feat_dim": feature_dimension
    }
    with open(save_file_path, "wb") as f:
        pickle.dump(data_to_save, f)
    return processed_nx_graphs, final_tagset, max_nodes_observed, feature_dimension

# --- 主数据集类 ---
class MyDataset:
    def __init__(self, config):
        self.config = config
        use_degree_as_tag = getattr(config.data, 'degree_as_tag', True)
        all_nx_graphs, self.tagset, max_nodes_data, feat_dim_data = \
            load_from_file(config, use_degree_as_tag)
        self.max_node_num = getattr(config.data, 'max_node_num', max_nodes_data)
        # 兼容 max_feat_dim 和 max_feat_num 两种字段
        if hasattr(config.data, 'max_feat_dim'):
            self.max_feat_num = getattr(config.data, 'max_feat_dim')
        elif hasattr(config.data, 'max_feat_num'):
            self.max_feat_num = getattr(config.data, 'max_feat_num')
        else:
            self.max_feat_num = len(self.tagset) if len(self.tagset) > 0 else feat_dim_data
        split_file = pathlib.Path(f"./datasets/{config.data.name}/train_test_classes.json")
        with open(split_file, "r") as f:
            class_splits = json.load(f)
            self.train_original_classes_set = set(map(int, class_splits["train"]))
            self.test_original_classes_set = set(map(int, class_splits["test"]))
        self.train_nx_graphs_internal = []
        train_orig_labels_list = []
        self.test_nx_graphs_internal = []
        test_orig_labels_list = []
        for nx_g in all_nx_graphs:
            original_label = nx_g.graph['label']
            if original_label in self.train_original_classes_set:
                self.train_nx_graphs_internal.append(nx_g)
                train_orig_labels_list.append(original_label)
            elif original_label in self.test_original_classes_set:
                self.test_nx_graphs_internal.append(nx_g)
                test_orig_labels_list.append(original_label)
        self.train_graphs = self.train_nx_graphs_internal
        self.test_graphs = self.test_nx_graphs_internal
        unique_train_sorted = sorted(list(self.train_original_classes_set))
        train_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_train_sorted)}
        self.train_labels_remapped = torch.LongTensor([train_remapper[l] for l in train_orig_labels_list])
        self.num_train_classes_remapped = len(unique_train_sorted)
        unique_test_sorted = sorted(list(self.test_original_classes_set))
        test_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_test_sorted)}
        self.test_labels_remapped = torch.LongTensor([test_remapper[l] for l in test_orig_labels_list])
        self.num_test_classes_remapped = len(unique_test_sorted)
        print(f"Converting training graphs to tensor (max_nodes={self.max_node_num}, target_feat_dim={self.max_feat_num})...")
        self.train_adjs, self.train_x = graphs_to_tensor(self.train_nx_graphs_internal, self.max_node_num, self.max_feat_num)
        print(f"Converting testing graphs to tensor (max_nodes={self.max_node_num}, target_feat_dim={self.max_feat_num})...")
        self.test_adjs, self.test_x = graphs_to_tensor(self.test_nx_graphs_internal, self.max_node_num, self.max_feat_num)
        self.train_indices_by_class = None
        self.test_indices_by_class = None
        if hasattr(self.config, "fsl_task"):
            # 构建类别到样本索引的映射
            self.train_indices_by_class = defaultdict(list)
            for i, remapped_label in enumerate(self.train_labels_remapped.tolist()):
                self.train_indices_by_class[remapped_label].append(i)
            self.test_indices_by_class = defaultdict(list)
            for i, remapped_label in enumerate(self.test_labels_remapped.tolist()):
                self.test_indices_by_class[remapped_label].append(i)
            # ----------- 确定性测试采样支持集与查询池 -----------
            # 固定种子，保证每次实验可复现
            rng = np.random.RandomState(42)
            K_shot = getattr(self.config.fsl_task, 'K_shot', 1)
            self.deterministic_test_support_indices_by_class = {}
            query_pool = []
            for cls, indices in self.test_indices_by_class.items():
                idxs = list(indices)
                rng.shuffle(idxs)
                self.deterministic_test_support_indices_by_class[cls] = idxs[:K_shot]
                query_pool.extend(idxs[K_shot:])
            rng.shuffle(query_pool)
            self.deterministic_test_query_pool_indices = query_pool
        else:
            print("No 'fsl_task' in config. Skipping FSL index preparation.")
    def __init__(self, config):
        self.config = config
        use_degree_as_tag = getattr(config.data, 'degree_as_tag', True)
        all_nx_graphs, self.tagset, max_nodes_data, feat_dim_data = \
            load_from_file(config, use_degree_as_tag)
        self.max_node_num = getattr(config.data, 'max_node_num', max_nodes_data)
        # 兼容 max_feat_dim 和 max_feat_num 两种字段
        if hasattr(config.data, 'max_feat_dim'):
            self.max_feat_num = getattr(config.data, 'max_feat_dim')
        elif hasattr(config.data, 'max_feat_num'):
            self.max_feat_num = getattr(config.data, 'max_feat_num')
        else:
            self.max_feat_num = len(self.tagset) if len(self.tagset) > 0 else feat_dim_data
        split_file = pathlib.Path(f"./datasets/{config.data.name}/train_test_classes.json")
        with open(split_file, "r") as f:
            class_splits = json.load(f)
            self.train_original_classes_set = set(map(int, class_splits["train"]))
            self.test_original_classes_set = set(map(int, class_splits["test"]))
        self.train_nx_graphs_internal = []
        train_orig_labels_list = []
        self.test_nx_graphs_internal = []
        test_orig_labels_list = []
        for nx_g in all_nx_graphs:
            original_label = nx_g.graph['label']
            if original_label in self.train_original_classes_set:
                self.train_nx_graphs_internal.append(nx_g)
                train_orig_labels_list.append(original_label)
            elif original_label in self.test_original_classes_set:
                self.test_nx_graphs_internal.append(nx_g)
                test_orig_labels_list.append(original_label)
        self.train_graphs = self.train_nx_graphs_internal
        self.test_graphs = self.test_nx_graphs_internal
        unique_train_sorted = sorted(list(self.train_original_classes_set))
        train_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_train_sorted)}
        self.train_labels_remapped = torch.LongTensor([train_remapper[l] for l in train_orig_labels_list])
        self.num_train_classes_remapped = len(unique_train_sorted)
        unique_test_sorted = sorted(list(self.test_original_classes_set))
        test_remapper = {orig_lbl: i for i, orig_lbl in enumerate(unique_test_sorted)}
        self.test_labels_remapped = torch.LongTensor([test_remapper[l] for l in test_orig_labels_list])
        self.num_test_classes_remapped = len(unique_test_sorted)
        print(f"Converting training graphs to tensor (max_nodes={self.max_node_num}, target_feat_dim={self.max_feat_num})...")
        self.train_adjs, self.train_x = graphs_to_tensor(self.train_nx_graphs_internal, self.max_node_num, self.max_feat_num)
        print(f"Converting testing graphs to tensor (max_nodes={self.max_node_num}, target_feat_dim={self.max_feat_num})...")
        self.test_adjs, self.test_x = graphs_to_tensor(self.test_nx_graphs_internal, self.max_node_num, self.max_feat_num)
        self.train_indices_by_class = None
        self.test_indices_by_class = None
        if hasattr(self.config, "fsl_task"):
            self.train_indices_by_class = defaultdict(list)
            for i, remapped_label in enumerate(self.train_labels_remapped.tolist()):
                self.train_indices_by_class[remapped_label].append(i)
            self.test_indices_by_class = defaultdict(list)
            for i, remapped_label in enumerate(self.test_labels_remapped.tolist()):
                self.test_indices_by_class[remapped_label].append(i)
            # ----------- 确定性测试采样支持集与查询池构建 -----------
            K_shot = getattr(self.config.fsl_task, 'K_shot', 1)
            rng = np.random.RandomState(42)
            self.deterministic_test_support_indices_by_class = {}
            self.deterministic_test_query_pool_indices = []
            for cls in range(self.num_test_classes_remapped):
                indices = list(self.test_indices_by_class[cls])
                rng.shuffle(indices)
                support = indices[:K_shot]
                query = indices[K_shot:]
                self.deterministic_test_support_indices_by_class[cls] = support
                self.deterministic_test_query_pool_indices.extend(query)
            rng.shuffle(self.deterministic_test_query_pool_indices)
        else:
            print("No 'fsl_task' in config. Skipping FSL index preparation.")

    def get_loaders(self):
        batch_size = getattr(self.config.data, 'batch_size', 32)
        num_workers = getattr(self.config.data, 'num_workers', 0)
        train_dataset = TensorDataset(self.train_x, self.train_adjs, self.train_labels_remapped)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = TensorDataset(self.test_x, self.test_adjs, self.test_labels_remapped)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader

    def _prepare_task_class_samples(self, all_indices_for_class_list, num_needed):
        if not all_indices_for_class_list:
            return [] 
        if len(all_indices_for_class_list) >= num_needed:
            return all_indices_for_class_list[:num_needed]
        num_repeats = (num_needed + len(all_indices_for_class_list) -1) // len(all_indices_for_class_list)
        return (all_indices_for_class_list * num_repeats)[:num_needed]

    def _sample_indices_for_task(self, indices_map_by_class, list_of_available_classes, N_way, K_shot, R_query):
        eligible_classes = [
            cls for cls in list_of_available_classes 
            if len(indices_map_by_class.get(cls, [])) >= K_shot 
        ]
        if len(eligible_classes) < N_way:
            return None, None
        chosen_remapped_class_labels = np.random.choice(eligible_classes, N_way, replace=False)
        task_support_indices = []
        task_query_indices = []
        for remapped_cls_label in chosen_remapped_class_labels:
            all_samples_for_this_remapped_class = list(indices_map_by_class[remapped_cls_label])
            np.random.shuffle(all_samples_for_this_remapped_class)
            support_for_this_class = self._prepare_task_class_samples(all_samples_for_this_remapped_class, K_shot)
            task_support_indices.extend(support_for_this_class)
            if len(all_samples_for_this_remapped_class) >= K_shot + R_query:
                query_for_this_class = all_samples_for_this_remapped_class[K_shot : K_shot + R_query]
            else:
                query_for_this_class = self._prepare_task_class_samples(all_samples_for_this_remapped_class, R_query)
            task_query_indices.extend(query_for_this_class)
        if len(task_support_indices) != N_way * K_shot or \
           len(task_query_indices) != N_way * R_query:
            return None, None
        return task_support_indices, task_query_indices

    def sample_one_task(self, is_train, N_way, K_shot, R_query, query_pool_start_index=None):
        """
        支持集和查询集均以张量形式返回，标签为真实remapped label。
        测试模式下N_way自动等于测试类别数，支持集和查询集为确定性采样。
        """
        if self.train_indices_by_class is None or self.test_indices_by_class is None:
            raise RuntimeError("Dataset not set up for FSL. Ensure 'fsl_task' in config.")
        if is_train:
            current_data_x, current_data_adjs = self.train_x, self.train_adjs
            current_indices_map = self.train_indices_by_class
            num_classes_in_set = self.num_train_classes_remapped
            if num_classes_in_set < N_way:
                raise ValueError(f"Not enough classes ({num_classes_in_set}) in train set for {N_way}-way task.")
            list_of_available_remapped_classes = list(current_indices_map.keys())
            task_support_indices, task_query_indices = None, None
            for _ in range(10):
                task_support_indices, task_query_indices = self._sample_indices_for_task(
                    current_indices_map, list_of_available_remapped_classes, N_way, K_shot, R_query
                )
                if task_support_indices and task_query_indices:
                    break
            if not task_support_indices or not task_query_indices:
                raise RuntimeError(f"Failed to sample a valid task after multiple retries. "
                                   "Check data distribution, N_way, K_shot, R_query.")
            support_x = current_data_x[task_support_indices]
            support_adj = current_data_adjs[task_support_indices]
            query_x = current_data_x[task_query_indices]
            query_adj = current_data_adjs[task_query_indices]
            support_labels = torch.arange(N_way, dtype=torch.long).repeat_interleave(K_shot)
            query_labels = torch.arange(N_way, dtype=torch.long).repeat_interleave(R_query)
            return {
                "support_set": {"x": support_x, "adj": support_adj, "label": support_labels},
                "query_set": {"x": query_x, "adj": query_adj, "label": query_labels},
                "append_count": 0
            }
        else:
            N_way = self.num_test_classes_remapped
            current_data_x, current_data_adjs = self.test_x, self.test_adjs
            support_indices = []
            support_labels = []
            for cls in range(N_way):
                indices = self.deterministic_test_support_indices_by_class[cls][:K_shot]
                support_indices.extend(indices)
                support_labels.extend([cls]*len(indices))
            support_x = current_data_x[support_indices]
            support_adj = current_data_adjs[support_indices]
            support_labels = torch.tensor(support_labels, dtype=torch.long)
            if query_pool_start_index is None:
                query_pool_start_index = 0
            query_indices = self.deterministic_test_query_pool_indices[query_pool_start_index:query_pool_start_index+R_query]
            query_x = current_data_x[query_indices]
            query_adj = current_data_adjs[query_indices]
            query_labels = self.test_labels_remapped[query_indices]
            return {
                "support_set": {"x": support_x, "adj": support_adj, "label": support_labels},
                "query_set": {"x": query_x, "adj": query_adj, "label": query_labels},
                "append_count": 0
            }

# --- 主数据加载入口 ---
def load_data(config, get_graph_list=False):
    dataset = MyDataset(config)
    if get_graph_list:
        return dataset.train_graphs, dataset.test_graphs
    else:
        return dataset.get_loaders()
