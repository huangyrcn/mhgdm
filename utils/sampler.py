import os
import time
import pickle
import math

import geoopt
import ml_collections
import numpy as np
import torch
import wandb

# Add numpy to safe globals for torch.load with weights_only=True
import numpy
torch.serialization.add_safe_globals([
    numpy.generic, numpy.ndarray,
    numpy.bool_, numpy.int_,
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
    numpy.float_, 
    numpy.float16, numpy.float32, numpy.float64,
    numpy.complex_,
    numpy.complex64, numpy.complex128,
    numpy.object_, numpy.str_, numpy.bytes_
])

import models.Encoders as Encoders
from utils.data_utils import MyDataset
from utils.manifolds_utils import get_manifold
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import (
    load_ckpt,
    load_data,  # 添加 load_data 导入
    load_seed,
    load_device,
    load_model_from_ckpt,
    load_ema_from_ckpt,
    load_sampling_fn,
    load_eval_settings,
    load_batch,
)
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import torch.nn.functional as F
from utils.graph_utils import node_flags


class Sampler(object):
    """
    使用预训练模型生成图的采样器类。
    """

    def __init__(self, config):

        self.config = config
        device_obj, device_str = load_device(
            device_setting=getattr(config, "device", "auto"),
            device_count=getattr(config, "device_count", 1)
        )
        self.device = device_obj
        config.device = device_str  # 更新config中的device设置

        # ---------- 加载分数网络模型 ----------
        self.independent = True
        score_ckpt = torch.load(config.score_ckpt_path, map_location=self.device, weights_only=False)
        self.configt = ml_collections.ConfigDict(score_ckpt["model_config"])
        
        # 加载分数网络模型
        self.mx = load_model_from_ckpt(self.device, score_ckpt["params_x"], score_ckpt["x_state_dict"])
        self.ma = load_model_from_ckpt(
            self.device, score_ckpt["params_adj"], score_ckpt["adj_state_dict"]
        )

        # 如果使用EMA，加载EMA权重
        if config.sample.use_ema:
            load_ema_from_ckpt(self.mx, score_ckpt["ema_x_state_dict"], self.configt.train.ema).copy_to(
                self.mx.parameters()
            )
            load_ema_from_ckpt(self.ma, score_ckpt["ema_adj_state_dict"], self.configt.train.ema).copy_to(
                self.ma.parameters()
            )
        
        # 设置分数网络为评估模式
        for p in self.mx.parameters():
            p.requires_grad = False
        for p in self.ma.parameters():
            p.requires_grad = False
        self.mx.eval()
        self.ma.eval()
        
        # 从训练配置加载随机种子
        load_seed(self.configt.seed)

        # ---------- 加载编码器模型 ----------
        # 检查是否有独立的编码器检查点路径
        if hasattr(config, 'encoder_ckpt_path') and config.encoder_ckpt_path:
            # 加载独立训练的编码器
            print(f"Loading encoder from: {config.encoder_ckpt_path}")
            encoder_ckpt = torch.load(config.encoder_ckpt_path, map_location=self.device, weights_only=False)
            
            encoder_checkpoint_config_dict = encoder_ckpt["model_config"]
            encoder_state_dict = encoder_ckpt["encoder_state_dict"]
            
            # 实例化编码器
            encoder_name = encoder_checkpoint_config_dict["model"]["encoder"]
            EncoderClass = getattr(Encoders, encoder_name)
            encoder_config_for_instantiation = ml_collections.ConfigDict(encoder_checkpoint_config_dict)
            
            # Create encoder with extracted parameters
            from utils.model_utils import extract_encoder_params
            encoder_params = extract_encoder_params(encoder_config_for_instantiation)
            self.encoder = EncoderClass(**encoder_params).to(self.device)
            self.encoder.load_state_dict(encoder_state_dict)
            
        else:
            raise ValueError("必须提供 encoder_ckpt_path 来加载编码器，不再支持从 AE checkpoint 加载")
        
        # 设置编码器为评估模式
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.to(self.device).eval()

        # ---------- 检查流形一致性 ----------
        encoder_manifold = getattr(self.encoder, "manifold", None)
        score_manifold_x = getattr(self.mx, "manifold", None)
        score_manifold_adj = getattr(self.ma, "manifold", None)
        
        # 验证所有模型使用相同的流形
        if encoder_manifold is not None and score_manifold_x is not None:
            if type(encoder_manifold) != type(score_manifold_x):
                raise ValueError(
                    f"编码器流形类型 {type(encoder_manifold)} 与分数网络X流形类型 {type(score_manifold_x)} 不匹配"
                )
            
            # 检查曲率参数（如果有）
            if hasattr(encoder_manifold, 'c') and hasattr(score_manifold_x, 'c'):
                if abs(encoder_manifold.c.item() - score_manifold_x.c.item()) > 1e-6:
                    raise ValueError(
                        f"编码器流形曲率 {encoder_manifold.c.item()} 与分数网络X流形曲率 {score_manifold_x.c.item()} 不匹配"
                    )
        
        if score_manifold_x is not None and score_manifold_adj is not None:
            if type(score_manifold_x) != type(score_manifold_adj):
                raise ValueError(
                    f"分数网络X流形类型 {type(score_manifold_x)} 与分数网络Adj流形类型 {type(score_manifold_adj)} 不匹配"
                )
        
        # 使用编码器的流形作为主流形
        self.manifold = encoder_manifold if encoder_manifold is not None else score_manifold_x
        
        print(f"采样器使用流形: {self.manifold}")

        # ---------- 加载数据和计算原型 ----------
        # 如果config中有dataloader就使用config中的，否则创建新的
        if hasattr(self.config, "dataloader") and self.config.dataloader is not None:
            self.dataloader = self.config.dataloader
        else:            # 使用MyDataset替代load_data以保持一致性
            dataset = MyDataset(self.configt.data, self.configt.fsl_task)
            _, self.dataloader = dataset.get_loaders()

        # ---------- 设置采样相关配置 ----------
        # 设置采样日志目录和文件夹名称
        self.log_folder_name, self.log_dir, _ = set_log(self.config, is_train=False)
        self.sampling_fn = load_sampling_fn(
            self.configt, config.sampler, config.sample, self.device, self.manifold
        )
        # 定义日志文件名
        self.log_name = f"{self.config.exp_name}-sample"
        # 初始化日志记录器实例
        self.logger = Logger(str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a")

        # 检查日志文件是否存在，如果不存在，则写入初始日志信息
        if not check_log(self.log_folder_name, self.log_name):
            self.logger.log(f"{self.log_name}")
            start_log(self.logger, self.configt)  # 记录启动配置
            train_log(self.logger, self.configt)  # 记录训练配置详情
        # 记录采样特定配置
        sample_log(self.logger, self.config)

    def sample(self, need_eval=True):
        """
        按batch采样，每个batch独立生成。
        Args:
            need_eval (bool): 是否评估生成图。
        Returns:
            dict or DataLoader: 评估指标 or 增强后的数据加载器。
        """


        gen_graph_list = []
        graph_ref_list = []

        # 存储k-augmented数据
        k_augment = self.config.sample.k_augment
        augmented_x_list = []
        augmented_adj_list = []
        augmented_labels_list = []

        with torch.no_grad():  # 采样阶段禁用梯度计算
            for r, batch in enumerate(self.dataloader):
                x_real, adj_real, labels = load_batch(batch, self.device)
                t_start = time.time()

                current_batch_size = adj_real.shape[0]  # 直接取当前batch大小
                shape_x = (
                    current_batch_size,
                    self.config.data.max_node_num,
                    self.config.data.max_feat_num,
                )
                shape_adj = (
                    current_batch_size,
                    self.config.data.max_node_num,
                    self.config.data.max_node_num,
                )

                # 为每个原始样本生成k个增强样本
                for _ in range(k_augment):
                    # 注意：sample方法仍使用原有逻辑，不涉及任务特定原型
                    # 如果需要基于任务的采样，请使用augment_task方法
                    x_gen, adj_gen = self.sampling_fn(
                        self.mx, self.ma, shape_x, shape_adj, labels, None  # 不使用原型
                    )

                    # 添加到增强数据集
                    augmented_x_list.append(x_gen)
                    augmented_adj_list.append(adj_gen)
                    augmented_labels_list.append(labels)

                # 添加原始数据到增强数据集
                augmented_x_list.append(x_real)
                augmented_adj_list.append(adj_real)
                augmented_labels_list.append(labels)

                self.logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

                # 保存生成的图和原始图（用于评估）
                if need_eval:
                    samples_int = quantize(adj_gen)
                    gen_graph_list.extend(adjs_to_graphs(samples_int, True))

                    adjs_real_int = quantize(adj_real)
                    graph_ref_list.extend(adjs_to_graphs(adjs_real_int, True))

        # 创建增强数据集的数据加载器
        augmented_x = torch.cat(augmented_x_list, dim=0)
        augmented_adj = torch.cat(augmented_adj_list, dim=0)
        augmented_labels = torch.cat(augmented_labels_list, dim=0)

        # 创建数据集和数据加载器
        from torch.utils.data import TensorDataset, DataLoader

        augmented_dataset = TensorDataset(augmented_x, augmented_adj, augmented_labels)
        batch_size = self.dataloader.batch_size if hasattr(self.dataloader, "batch_size") else 32
        augmented_dataloader = DataLoader(
            augmented_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=(
                self.dataloader.num_workers if hasattr(self.dataloader, "num_workers") else 0
            ),
        )

        # 返回增强的数据加载器（如果不需要评估）
        if not need_eval:
            self.logger.log(
                f"Created k-augmented dataloader with k={k_augment}, total samples: {len(augmented_dataset)}"
            )
            return augmented_dataloader

        # --------- 评估 ---------
        methods, kernels = load_eval_settings()
        result_dict = eval_graph_list(
            graph_ref_list, gen_graph_list, methods=methods, kernels=kernels
        )
        result_dict["mean"] = (
            result_dict["degree"] + result_dict["cluster"] + result_dict["orbit"]
        ) / 3
        print(result_dict)
        self.logger.log(
            f"MMD_full {result_dict}"
            f"\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-"
            f"X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}"
            f"\n{self.config.run_name}",
            verbose=False,
        )

        self.logger.log("=" * 100)
        if self.independent:
            wandb.log(result_dict, commit=True)

        # 同时返回评估结果和增强的数据加载器
        result_dict["augmented_dataloader"] = augmented_dataloader
        return result_dict

    def augment_task(self, task, k_augment=None):
        """
        增强任务数据：基于支持集计算原型，然后使用这些原型指导生成新样本。
        
        根据您的设计，采样函数接收一个任务，然后基于支持集计算原型，
        再基于这些原型生成新的任务，并返回增强后的任务。
        
        Args:
            task: 从encoder_trainer获得的任务字典，包含：
                - support_set: {"x": tensor, "adj": tensor, "label": tensor}  
                - query_set: {"x": tensor, "adj": tensor, "label": tensor}
                - append_count: int
            k_augment: 每个原始支持样本生成的增强样本数量，默认使用config中的值
        
        Returns:
            augmented_task: 增强后的任务字典，支持集包含原始样本+生成样本
        """
        if k_augment is None:
            k_augment = getattr(self.config.sample, 'k_augment', 1)
        
        # 提取支持集数据
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_labels = task["support_set"]["label"].to(self.device)
        
        # 查询集保持不变
        query_x = task["query_set"]["x"]
        query_adj = task["query_set"]["adj"]
        query_labels = task["query_set"]["label"]
        
        # 基于支持集计算任务特定的原型
        task_prototypes = self._compute_task_prototypes(support_x, support_adj, support_labels)
        
        # 为每个支持样本生成k_augment个增强样本
        augmented_x_list = [support_x]  # 包含原始支持集
        augmented_adj_list = [support_adj]
        augmented_labels_list = [support_labels]
        
        with torch.no_grad():
            for _ in range(k_augment):
                # 为每个支持样本生成增强样本
                batch_size = support_x.shape[0]
                shape_x = (batch_size, self.configt.data.max_node_num, self.configt.data.max_feat_num)
                shape_adj = (batch_size, self.configt.data.max_node_num, self.configt.data.max_node_num)
                
                # 使用采样函数生成新样本，传入支持集标签和任务特定原型
                x_gen, adj_gen = self.sampling_fn(
                    self.mx, self.ma, shape_x, shape_adj, support_labels, task_prototypes
                )
                
                augmented_x_list.append(x_gen)
                augmented_adj_list.append(adj_gen)
                augmented_labels_list.append(support_labels)  # 保持相同的标签
        
        # 合并所有增强数据
        augmented_support_x = torch.cat(augmented_x_list, dim=0)
        augmented_support_adj = torch.cat(augmented_adj_list, dim=0)
        augmented_support_labels = torch.cat(augmented_labels_list, dim=0)
        
        # 构建增强后的任务
        augmented_task = {
            "support_set": {
                "x": augmented_support_x,
                "adj": augmented_support_adj,
                "label": augmented_support_labels
            },
            "query_set": {
                "x": query_x,
                "adj": query_adj,
                "label": query_labels
            },
            "append_count": task.get("append_count", 0)
        }
        
        self.logger.log(
            f"Task augmented with task-specific prototypes: original support set size {support_x.shape[0]} -> "
            f"augmented support set size {augmented_support_x.shape[0]} (k_augment={k_augment}). "
            f"Computed {task_prototypes.shape[0]} task-specific prototypes."
        )
        
        return augmented_task

    def _compute_task_prototypes(self, support_x, support_adj, support_labels):
        """
        基于支持集计算任务特定的原型。
        
        Args:
            support_x: 支持集特征 [B, N, F]
            support_adj: 支持集邻接矩阵 [B, N, N]
            support_labels: 支持集标签 [B]
        
        Returns:
            task_prototypes: 任务特定原型 [num_classes, D]
        """
        # 计算支持集的嵌入
        node_masks = torch.stack([node_flags(adj) for adj in support_adj])
        
        with torch.no_grad():
            posterior = self.encoder(support_x, support_adj, node_masks)
            embeddings = posterior.mode()  # [B, N, D] 或 [B, D]
            
            # 确保嵌入是图级别的表示
            if embeddings.dim() == 3:
                # 如果是节点级别的嵌入，需要聚合为图级别
                if embeddings.size(1) == 1:
                    embeddings = embeddings.squeeze(1)  # [B, D]
                else:
                    # 使用平均池化 + 最大池化连接聚合节点特征（与encoder_trainer.py一致）
                    valid_nodes = node_masks.float().unsqueeze(-1)  # [B, N, 1]
                    masked_embeddings = embeddings * valid_nodes  # [B, N, D]
                    
                    # 平均池化
                    mean_emb = masked_embeddings.sum(dim=1) / valid_nodes.sum(dim=1)  # [B, D]
                    
                    # 最大池化 - 先将无效节点设为很小的值
                    masked_embeddings_for_max = masked_embeddings.clone()
                    invalid_mask = (valid_nodes == 0).expand_as(masked_embeddings)
                    masked_embeddings_for_max[invalid_mask] = float('-inf')
                    max_emb = masked_embeddings_for_max.max(dim=1).values  # [B, D]
                    
                    # 连接平均池化和最大池化结果
                    embeddings = torch.cat([mean_emb, max_emb], dim=-1)  # [B, 2*D]
            
            # 如果编码器使用流形，将嵌入映射到切空间
            if self.encoder.manifold is not None:
                embeddings = self.encoder.manifold.logmap0(embeddings)
        
        # 按标签分组计算原型
        unique_labels = torch.unique(support_labels)
        task_prototypes = []
        
        for label in unique_labels:
            mask = (support_labels == label)
            if mask.sum() > 0:
                # 计算该类别的原型（取平均）
                class_embeddings = embeddings[mask]
                class_prototype = class_embeddings.mean(dim=0, keepdim=True)  # [1, D]
                task_prototypes.append(class_prototype)
        
        # 返回任务特定原型
        if task_prototypes:
            task_prototypes = torch.cat(task_prototypes, dim=0)  # [num_classes, D]
        else:
            # 如果没有计算出原型，创建一个默认的零原型
            # 使用嵌入维度作为原型维度
            emb_dim = embeddings.shape[-1] if embeddings.numel() > 0 else 64
            task_prototypes = torch.zeros(1, emb_dim, device=support_x.device)
        
        return task_prototypes


class GraphSampler(object):
    """
    专门用于基于任务采样的图采样器类。
    只包含必要的任务增强功能，不包含全局原型相关的功能。
    """

    def __init__(self, config, dataset=None):
        """
        初始化GraphSampler
        
        Args:
            config: 配置对象，包含模型和采样配置
            dataset: 可选的数据集对象，如果不提供则从config创建
        """
        self.config = config
        device_obj, device_str = load_device(
            device_setting=getattr(config, "device", "auto"),
            device_count=getattr(config, "device_count", 1)
        )
        self.device = device_obj
        config.device = device_str

        # ---------- 加载分数网络模型 ----------
        score_ckpt = torch.load(config.score_ckpt_path, map_location=self.device, weights_only=False)
        self.configt = ml_collections.ConfigDict(score_ckpt["model_config"])
        
        # 加载分数网络模型
        self.mx = load_model_from_ckpt(self.device, score_ckpt["params_x"], score_ckpt["x_state_dict"])
        self.ma = load_model_from_ckpt(
            self.device, score_ckpt["params_adj"], score_ckpt["adj_state_dict"]
        )

        # 如果使用EMA，加载EMA权重
        if config.sample.use_ema:
            load_ema_from_ckpt(self.mx, score_ckpt["ema_x_state_dict"], self.configt.train.ema).copy_to(
                self.mx.parameters()
            )
            load_ema_from_ckpt(self.ma, score_ckpt["ema_adj_state_dict"], self.configt.train.ema).copy_to(
                self.ma.parameters()
            )
        
        # 设置分数网络为评估模式
        for p in self.mx.parameters():
            p.requires_grad = False
        for p in self.ma.parameters():
            p.requires_grad = False
        self.mx.eval()
        self.ma.eval()

        # ---------- 加载编码器模型 ----------
        if hasattr(config, 'encoder_ckpt_path') and config.encoder_ckpt_path:
            encoder_ckpt = torch.load(config.encoder_ckpt_path, map_location=self.device, weights_only=False)
            
            if "encoder_config" in encoder_ckpt:
                encoder_checkpoint_config_dict = encoder_ckpt["encoder_config"]
                encoder_state_dict = encoder_ckpt["encoder_state_dict"]
            elif "model_config" in encoder_ckpt:
                encoder_checkpoint_config_dict = encoder_ckpt["model_config"]
                encoder_state_dict = encoder_ckpt["encoder_state_dict"]
            else:
                raise ValueError("无法从检查点中找到编码器配置")
            
            # 确定编码器类名
            if "encoder" in encoder_checkpoint_config_dict:
                encoder_name = encoder_checkpoint_config_dict["encoder"]["model_class"]
            elif "model" in encoder_checkpoint_config_dict:
                encoder_name = encoder_checkpoint_config_dict["model"]["encoder"]
            else:
                raise ValueError("无法从检查点配置中确定编码器类名")
            
            # 实例化编码器
            EncoderClass = getattr(Encoders, encoder_name)
            encoder_config_for_instantiation = ml_collections.ConfigDict(encoder_checkpoint_config_dict)
            
            from utils.model_utils import extract_encoder_params
            encoder_params = extract_encoder_params(encoder_config_for_instantiation)
            self.encoder = EncoderClass(**encoder_params).to(self.device)
            self.encoder.load_state_dict(encoder_state_dict)
        else:
            raise ValueError("必须提供 encoder_ckpt_path 来加载编码器")
        
        # 设置编码器为评估模式
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.to(self.device).eval()

        # ---------- 检查流形一致性 ----------
        encoder_manifold = getattr(self.encoder, "manifold", None)
        score_manifold_x = getattr(self.mx, "manifold", None)
        score_manifold_adj = getattr(self.ma, "manifold", None)
        
        if encoder_manifold is not None and score_manifold_x is not None:
            if type(encoder_manifold) != type(score_manifold_x):
                raise ValueError(
                    f"编码器流形类型 {type(encoder_manifold)} 与分数网络X流形类型 {type(score_manifold_x)} 不匹配"
                )
        
        self.manifold = encoder_manifold if encoder_manifold is not None else score_manifold_x

        # ---------- 设置采样相关配置 ----------
        # 如果有数据集就使用，否则创建一个简单的配置用于采样函数
        self.dataset = dataset
        if dataset is not None:
            # 设置采样日志
            self.log_folder_name, self.log_dir, _ = set_log(self.config, is_train=False)
            self.log_name = f"{self.config.exp_name}-task-sample"
            self.logger = Logger(str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a")
        else:
            # 简单的日志设置
            self.logger = None
        
        # 加载采样函数
        self.sampling_fn = load_sampling_fn(
            self.configt, config.sampler, config.sample, self.device, self.manifold
        )

    def augment_task(self, task, k_augment=None):
        """
        增强任务数据：基于支持集计算原型，然后使用这些原型指导生成新样本。
        
        Args:
            task: 任务字典，包含：
                - support_set: {"x": tensor, "adj": tensor, "label": tensor}  
                - query_set: {"x": tensor, "adj": tensor, "label": tensor}
                - append_count: int (可选)
            k_augment: 每个原始支持样本生成的增强样本数量，默认使用config中的值
        
        Returns:
            augmented_task: 增强后的任务字典，支持集包含原始样本+生成样本
        """
        if k_augment is None:
            k_augment = getattr(self.config.sample, 'k_augment', 1)
        
        # 提取支持集数据
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_labels = task["support_set"]["label"].to(self.device)
        
        # 查询集保持不变
        query_x = task["query_set"]["x"]
        query_adj = task["query_set"]["adj"]
        query_labels = task["query_set"]["label"]
        
        # 基于支持集计算任务特定的原型
        task_prototypes = self._compute_task_prototypes(support_x, support_adj, support_labels)
        
        # 为每个支持样本生成k_augment个增强样本
        augmented_x_list = [support_x]  # 包含原始支持集
        augmented_adj_list = [support_adj]
        augmented_labels_list = [support_labels]
        
        with torch.no_grad():
            for _ in range(k_augment):
                # 为每个支持样本生成增强样本
                batch_size = support_x.shape[0]
                shape_x = (batch_size, self.configt.data.max_node_num, self.configt.data.max_feat_num)
                shape_adj = (batch_size, self.configt.data.max_node_num, self.configt.data.max_node_num)
                
                # 使用采样函数生成新样本，传入支持集标签和任务特定原型
                x_gen, adj_gen = self.sampling_fn(
                    self.mx, self.ma, shape_x, shape_adj, support_labels, task_prototypes
                )
                
                augmented_x_list.append(x_gen)
                augmented_adj_list.append(adj_gen)
                augmented_labels_list.append(support_labels)  # 保持相同的标签
        
        # 合并所有增强数据
        augmented_support_x = torch.cat(augmented_x_list, dim=0)
        augmented_support_adj = torch.cat(augmented_adj_list, dim=0)
        augmented_support_labels = torch.cat(augmented_labels_list, dim=0)
        
        # 构建增强后的任务
        augmented_task = {
            "support_set": {
                "x": augmented_support_x,
                "adj": augmented_support_adj,
                "label": augmented_support_labels
            },
            "query_set": {
                "x": query_x,
                "adj": query_adj,
                "label": query_labels
            },
            "append_count": task.get("append_count", 0)
        }
        
        if self.logger:
            self.logger.log(
                f"Task augmented with task-specific prototypes: original support set size {support_x.shape[0]} -> "
                f"augmented support set size {augmented_support_x.shape[0]} (k_augment={k_augment}). "
                f"Computed {task_prototypes.shape[0]} task-specific prototypes."
            )
        
        return augmented_task

    def _compute_task_prototypes(self, support_x, support_adj, support_labels):
        """
        基于支持集计算任务特定的原型。
        
        Args:
            support_x: 支持集特征 [B, N, F]
            support_adj: 支持集邻接矩阵 [B, N, N]
            support_labels: 支持集标签 [B]
        
        Returns:
            task_prototypes: 任务特定原型 [num_classes, D]
        """
        # 计算支持集的嵌入
        node_masks = torch.stack([node_flags(adj) for adj in support_adj])
        
        with torch.no_grad():
            posterior = self.encoder(support_x, support_adj, node_masks)
            embeddings = posterior.mode()  # [B, N, D] 或 [B, D]
            
            # 确保嵌入是图级别的表示
            if embeddings.dim() == 3:
                # 如果是节点级别的嵌入，需要聚合为图级别
                if embeddings.size(1) == 1:
                    embeddings = embeddings.squeeze(1)  # [B, D]
                else:
                    # 使用平均池化 + 最大池化连接聚合节点特征（与encoder_trainer.py一致）
                    valid_nodes = node_masks.float().unsqueeze(-1)  # [B, N, 1]
                    masked_embeddings = embeddings * valid_nodes  # [B, N, D]
                    
                    # 平均池化
                    mean_emb = masked_embeddings.sum(dim=1) / valid_nodes.sum(dim=1)  # [B, D]
                    
                    # 最大池化 - 先将无效节点设为很小的值
                    masked_embeddings_for_max = masked_embeddings.clone()
                    invalid_mask = (valid_nodes == 0).expand_as(masked_embeddings)
                    masked_embeddings_for_max[invalid_mask] = float('-inf')
                    max_emb = masked_embeddings_for_max.max(dim=1).values  # [B, D]
                    
                    # 连接平均池化和最大池化结果
                    embeddings = torch.cat([mean_emb, max_emb], dim=-1)  # [B, 2*D]
            
            # 如果编码器使用流形，将嵌入映射到切空间
            if self.encoder.manifold is not None:
                embeddings = self.encoder.manifold.logmap0(embeddings)
        
        # 按标签分组计算原型
        unique_labels = torch.unique(support_labels)
        task_prototypes = []
        
        for label in unique_labels:
            mask = (support_labels == label)
            if mask.sum() > 0:
                # 计算该类别的原型（取平均）
                class_embeddings = embeddings[mask]
                class_prototype = class_embeddings.mean(dim=0, keepdim=True)  # [1, D]
                task_prototypes.append(class_prototype)
        
        # 返回任务特定原型
        if task_prototypes:
            task_prototypes = torch.cat(task_prototypes, dim=0)  # [num_classes, D]
        else:
            # 如果没有计算出原型，创建一个默认的零原型
            # 使用嵌入维度作为原型维度
            emb_dim = embeddings.shape[-1] if embeddings.numel() > 0 else 64
            task_prototypes = torch.zeros(1, emb_dim, device=support_x.device)
        
        return task_prototypes
