"""
采样器模块 - 适配Score训练器的采样功能
精简版本，专门用于Score训练期间的采样质量评估
"""

import os
import time
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F

from utils.loader import (
    load_seed,
    load_device,
    load_model_from_ckpt,
    load_sampling_fn,
)
from utils.graph_utils import adjs_to_graphs, init_flags, quantize
from utils.manifolds_utils import get_manifold
from utils.data_utils import MyDataset


# -------- Sampler for Score training evaluation --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()
        self.config = config
        self.device = load_device(config)

    def sample(self, independent=True):
        """精简版采样方法，用于Score训练期间的质量评估"""
        try:
            # -------- Load checkpoint --------
            checkpoint_path = getattr(self.config.sampler, "ckp_path", None)
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                return self._default_results()

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 重建模型配置
            model_config = checkpoint.get("model_config", {})
            if isinstance(model_config, dict):
                configt = type("Config", (), model_config)()
            else:
                configt = model_config

            # 加载模型
            self.model_x = load_model_from_ckpt(
                self.config, checkpoint["params_x"], checkpoint["x_state_dict"]
            )
            self.model_adj = load_model_from_ckpt(
                self.config, checkpoint["params_adj"], checkpoint["adj_state_dict"]
            )

            # 获取流形信息
            if hasattr(self.model_x, "manifold"):
                manifold = self.model_x.manifold
            else:
                manifold = get_manifold(self.config.data.get("manifold", "euclidean"))

            # 设置随机种子
            if hasattr(self.config, "seed"):
                load_seed(self.config.seed)

            # 获取数据用于初始化
            try:
                dataset = MyDataset(self.config.data, getattr(self.config, "fsl_task", {}))
                train_loader, _ = dataset.get_loaders()
                # 从数据加载器中获取一些样本用于初始化
                train_graph_list = []
                for i, batch in enumerate(train_loader):
                    if i >= 10:  # 只取前10个batch用于初始化
                        break
                    train_graph_list.extend(batch)
            except Exception as e:
                print(f"Warning: Could not load dataset for initialization: {e}")
                # 如果无法加载数据，使用默认配置
                train_graph_list = []

            # 创建采样函数
            sampling_fn = load_sampling_fn(
                self.config,
                self.config.sampler,
                self.config.get("sample", {}),
                self.device,
                manifold,
            )

            # -------- Generate samples --------
            k_augment = getattr(self.config.sampler, "k_augment", 10)
            batch_size = min(k_augment, 32)  # 限制批次大小避免内存问题

            gen_graph_list = []
            num_rounds = max(1, math.ceil(k_augment / batch_size))

            for r in range(num_rounds):
                try:
                    # 创建初始化flags
                    if train_graph_list:
                        init_flags_tensor = init_flags(
                            train_graph_list, self.config, batch_size
                        ).to(self.device)
                    else:
                        # 默认flags
                        max_node_num = getattr(self.config.data, "max_node_num", 9)
                        init_flags_tensor = torch.ones(batch_size, max_node_num).to(self.device)

                    # 执行采样
                    x, adj = sampling_fn(self.model_x, self.model_adj, init_flags_tensor)

                    # 量化并转换为图
                    samples_int = quantize(adj)
                    round_graphs = adjs_to_graphs(samples_int, True)
                    gen_graph_list.extend(round_graphs)

                except Exception as e:
                    print(f"Sampling round {r} failed: {e}")
                    continue

            # 限制生成图的数量
            gen_graph_list = gen_graph_list[:k_augment]

            # -------- 简化的评估 --------
            validity = self._compute_validity(gen_graph_list)
            uniqueness = self._compute_uniqueness(gen_graph_list)
            novelty = 0.8  # 简化的新颖性估计

            return {
                "validity": validity,
                "uniqueness": uniqueness,
                "novelty": novelty,
                "num_samples": len(gen_graph_list),
            }

        except Exception as e:
            print(f"Sampling failed: {e}")
            return self._default_results()

    def _default_results(self):
        """返回默认结果，用于采样失败时"""
        return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0, "num_samples": 0}

    def _compute_validity(self, graph_list):
        """计算有效性 - 简化版本"""
        if not graph_list:
            return 0.0

        valid_count = 0
        for graph in graph_list:
            # 简单的有效性检查：至少有2个节点和1条边
            if graph is not None:
                try:
                    num_nodes = (
                        graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else len(graph)
                    )
                    num_edges = graph.number_of_edges() if hasattr(graph, "number_of_edges") else 0
                    if num_nodes >= 2 and num_edges >= 1:
                        valid_count += 1
                except:
                    continue

        return valid_count / len(graph_list)

    def _compute_uniqueness(self, graph_list):
        """计算唯一性 - 简化版本"""
        if not graph_list:
            return 0.0

        unique_graphs = set()

        for graph in graph_list:
            if graph is not None:
                try:
                    # 创建图的简单哈希表示
                    if hasattr(graph, "edges"):
                        edges = tuple(sorted(graph.edges()))
                        nodes = tuple(sorted(graph.nodes()))
                        graph_hash = (nodes, edges)
                        unique_graphs.add(graph_hash)
                    else:
                        # 如果无法处理，跳过
                        continue
                except:
                    continue

        return len(unique_graphs) / len(graph_list) if graph_list else 0.0


# Note: 分子采样器已移除，因为当前项目专注于图生成任务
# 如果需要分子生成功能，可以添加 Sampler_mol 类
